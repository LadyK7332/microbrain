"""Two-window PySide6 engineering dashboard for MicroBrain."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from microbrain.ui.frontend_common import (
    TranscriptWriter,
    UIMessage,
    extract_evidence_refs,
    extract_text_and_channels,
    flatten_mapping,
    load_display_labels,
    safe_json,
    should_show_in_conversation,
)
from microbrain.ui.dashboard.config_catalog import scan_repo

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

UI_POLL_MS = 50
MAX_MESSAGES_PER_POLL = 120
MAX_RAW_LINES = 1500
MAX_TRACE_GROUPS = 120
MAX_EVIDENCE_ROWS = 400
VISION_OVERLAY_LINE_WIDTH = 2
LOG_TAIL_POLL_MS = 500
LOG_TAIL_INITIAL_BYTES = 65536
DEFAULT_SCREEN_MARGIN_PX = 20

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

TRACE_STAGE_PREFIXES = (
    ("input/", "Input"),
    ("percept/", "Perception"),
    ("context/", "Context"),
    ("pattern/", "Pattern"),
    ("hypothesis/", "Hypothesis"),
    ("thought/", "Thought"),
    ("release/", "Release"),
    ("reason/", "Action/Reason"),
    ("act/", "Output/Action"),
)


def _pretty(value: object) -> str:
    try:
        return json.dumps(safe_json(value), ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return repr(value)


def _summary(value: object, limit: int = 180) -> str:
    if isinstance(value, Mapping):
        text = str(value.get("text") or value.get("kind") or value.get("selected_action") or "")
        if not text:
            text = json.dumps(safe_json(value), ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _stage(topic: str) -> str:
    for prefix, label in TRACE_STAGE_PREFIXES:
        if topic.startswith(prefix):
            return label
    return "Bus"


class VisionCanvas(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 300)
        self._pixmap = QPixmap()
        self._overlays: list[dict[str, Any]] = []
        self._source_size = (0, 0)
        self._label = "No vision frame yet"

    def set_frame(self, path: str, width: int = 0, height: int = 0) -> None:
        candidate = Path(path)
        pixmap = QPixmap(str(candidate)) if candidate.exists() else QPixmap()
        if not pixmap.isNull():
            self._pixmap = pixmap
            self._source_size = (width or pixmap.width(), height or pixmap.height())
            self._label = candidate.name
            self.update()

    def set_overlays(self, overlays: list[dict[str, Any]]) -> None:
        self._overlays = overlays[-32:]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        if self._pixmap.isNull():
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignCenter, self._label)
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x0 = (self.width() - scaled.width()) // 2
        y0 = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x0, y0, scaled)
        sw, sh = self._source_size
        if sw <= 0 or sh <= 0:
            return
        sx, sy = scaled.width() / sw, scaled.height() / sh
        pen = QPen(self.palette().highlight().color(), VISION_OVERLAY_LINE_WIDTH)
        painter.setPen(pen)
        for item in self._overlays:
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                bx, by, bw, bh = [float(v) for v in bbox[:4]]
            except Exception:
                continue
            if max(abs(bx), abs(by), abs(bw), abs(bh)) <= 1.5:
                bx, bw = bx * sw, bw * sw
                by, bh = by * sh, bh * sh
            rx, ry, rw, rh = x0 + bx * sx, y0 + by * sy, bw * sx, bh * sy
            painter.drawRect(int(rx), int(ry), int(rw), int(rh))
            label = str(item.get("label") or "object")
            conf = item.get("confidence")
            if conf is not None:
                try:
                    label += f" {float(conf):.2f}"
                except Exception:
                    pass
            painter.drawText(int(rx) + 3, int(ry) + 14, label)
            painter.drawLine(self.width() // 2, self.height() // 2, int(rx + rw / 2), int(ry + rh / 2))


class BodyMapWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(280, 300)
        self._last: dict[str, str] = {}

    def update_event(self, msg: UIMessage) -> None:
        family = next((p for p in ("touch/", "motor/", "proprio/") if msg.topic.startswith(p)), None)
        if family:
            self._last[family.rstrip("/")] = f"{msg.topic}: {_summary(msg.payload, 90)}"
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().base())
        pen = QPen(self.palette().text().color(), 2)
        p.setPen(pen)
        cx, top = self.width() // 2, 42
        p.drawEllipse(cx - 18, top, 36, 36)
        p.drawLine(cx, top + 36, cx, top + 150)
        p.drawLine(cx, top + 65, cx - 65, top + 105)
        p.drawLine(cx, top + 65, cx + 65, top + 105)
        p.drawLine(cx, top + 150, cx - 45, top + 225)
        p.drawLine(cx, top + 150, cx + 45, top + 225)
        p.drawText(10, 20, "Body / proprioception map")
        y = max(top + 250, 300)
        if not self._last:
            p.drawText(10, y, "Awaiting touch / motor / proprioception telemetry")
        else:
            for family, text in sorted(self._last.items()):
                p.drawText(10, y, f"{family}: {text[:80]}")
                y += 18


class LogTailWidget(QPlainTextEdit):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(MAX_RAW_LINES)
        self._path = path
        self._offset = 0
        self._initialized = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_file)
        self._timer.start(LOG_TAIL_POLL_MS)

    def _poll_file(self) -> None:
        try:
            if not self._path.exists():
                return
            size = self._path.stat().st_size
            if size < self._offset:
                self._offset = 0
            if not self._initialized:
                self._offset = max(0, size - LOG_TAIL_INITIAL_BYTES)
                self._initialized = True
            if size <= self._offset:
                return
            with self._path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
                self._offset = handle.tell()
            if chunk:
                self.appendPlainText(chunk.rstrip("\n"))
        except Exception:
            pass


class PresenceWindow(QMainWindow):
    def __init__(self, controller: "DashboardController") -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("MicroBrain — Presence / Perception")
        root = QWidget()
        layout = QVBoxLayout(root)
        self.status = QLabel("body> starting | pulse> waiting")
        layout.addWidget(self.status)
        top = QSplitter(Qt.Horizontal)
        self.vision = VisionCanvas()
        self.body = BodyMapWidget()
        top.addWidget(self.vision)
        top.addWidget(self.body)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 1)
        layout.addWidget(top, 3)
        bottom = QSplitter(Qt.Horizontal)
        self.conversation = QPlainTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setPlaceholderText("Conversation / visible interaction")
        self.process_log = QPlainTextEdit()
        self.process_log.setReadOnly(True)
        self.process_log.setPlaceholderText("Process / component trace")
        self.process_log.setMaximumBlockCount(MAX_RAW_LINES)
        bottom.addWidget(self.conversation)
        bottom.addWidget(self.process_log)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)
        layout.addWidget(bottom, 2)
        send_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type here…")
        self.send_button = QPushButton("Send")
        send_row.addWidget(self.input, 1)
        send_row.addWidget(self.send_button)
        layout.addLayout(send_row)
        self.setCentralWidget(root)
        self.send_button.clicked.connect(self._submit)
        self.input.returnPressed.connect(self._submit)

    def _submit(self) -> None:
        text = self.input.text().strip()
        self.input.clear()
        if text:
            self.controller.submit_text(text)

    def append_conversation(self, line: str) -> None:
        self.conversation.appendPlainText(line)

    def append_process(self, line: str) -> None:
        self.process_log.appendPlainText(line)

    def update_pressure(self, payload: Mapping[str, Any]) -> None:
        body = payload.get("body", {}) if isinstance(payload.get("body"), Mapping) else {}
        pulse = payload.get("pulse", {}) if isinstance(payload.get("pulse"), Mapping) else {}
        self.status.setText(
            "body> pwr {pwr} chg:{chg} sleep:{sleep} mem:{mem} read:{read} | "
            "pulse> sal {sal:.2f} dop {dop:.2f} bored {bored:.2f} cur {cur:.2f} "
            "expr {expr:.2f} think {think:.2f} | {intent}/{state}".format(
                pwr=body.get("power_mode", "awake"), chg="on" if body.get("charging") else "off",
                sleep="on" if body.get("sleep") else "off", mem=body.get("memory_composer", "off"),
                read=body.get("read_sidecar", "off"), sal=float(pulse.get("salience", 0) or 0),
                dop=float(pulse.get("reward", 0) or 0), bored=float(pulse.get("boredom", 0) or 0),
                cur=float(pulse.get("curiosity", 0) or 0), expr=float(pulse.get("expression", 0) or 0),
                think=float(pulse.get("thought_pressure", 0) or 0), intent=pulse.get("thought_intent", "idle"),
                state=pulse.get("thought_status", "idle"),
            )
        )

    def process(self, msg: UIMessage) -> None:
        self.body.update_event(msg)
        payload = msg.payload if isinstance(msg.payload, Mapping) else {}
        if msg.topic == "percept/vision":
            ref = str(payload.get("data_ref") or "")
            if ref:
                self.vision.set_frame(ref, int(payload.get("width") or 0), int(payload.get("height") or 0))
        overlays: list[dict[str, Any]] = []
        if msg.topic == "vision/percept_commit":
            overlays = [{"bbox": payload.get("crop_box"), "label": payload.get("resolved_label"), "confidence": payload.get("max_stability")}]
            ref = str(payload.get("frame_ref") or "")
            if ref:
                self.vision.set_frame(ref)
        elif msg.topic == "percept/vision/features":
            raw = payload.get("objects") or payload.get("features") or []
            overlays = [dict(x) for x in raw if isinstance(x, Mapping)] if isinstance(raw, list) else []
        elif msg.topic == "vision/object_delta":
            ref = str(payload.get("image_ref") or "")
            if ref:
                self.vision.set_frame(ref)
            for delta in payload.get("deltas", []) if isinstance(payload.get("deltas"), list) else []:
                cur = delta.get("current") if isinstance(delta, Mapping) else None
                if isinstance(cur, Mapping):
                    overlays.append(dict(cur))
        if overlays:
            self.vision.set_overlays(overlays)


class EngineeringWindow(QMainWindow):
    def __init__(self, controller: "DashboardController") -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("MicroBrain — Engineering / Internal")
        self.trace = QTreeWidget()
        self.trace.setHeaderLabels(["Stage / topic", "Source", "Summary"])
        self.trace.itemSelectionChanged.connect(self._trace_selected)
        self.setCentralWidget(self.trace)
        self._groups: dict[str, QTreeWidgetItem] = {}
        self._messages: dict[int, UIMessage] = {}

        self.raw = QPlainTextEdit(); self.raw.setReadOnly(True); self.raw.setMaximumBlockCount(MAX_RAW_LINES)
        self.detail = QPlainTextEdit(); self.detail.setReadOnly(True)
        self.evidence = QTreeWidget(); self.evidence.setHeaderLabels(["Type", "Reference", "From"])
        self.evidence.itemDoubleClicked.connect(self._open_evidence)
        self.organs = QPlainTextEdit(); self.organs.setReadOnly(True)
        self.runtime_log = LogTailWidget(Path(self.controller.bridge.memdir) / "logs" / "microbrain.log")
        self.tuning_tabs = self._build_tuning_tabs()
        self._dock("Raw event bus", self.raw, Qt.BottomDockWidgetArea)
        self._dock("Runtime log", self.runtime_log, Qt.BottomDockWidgetArea)
        self._dock("Selected event", self.detail, Qt.RightDockWidgetArea)
        self._dock("Evidence links", self.evidence, Qt.RightDockWidgetArea)
        self._dock("DDNA / tuning / laws", self.tuning_tabs, Qt.LeftDockWidgetArea)
        self._dock("Organ / bus status", self.organs, Qt.BottomDockWidgetArea)
        self._load_config_catalog()

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> None:
        dock = QDockWidget(title, self); dock.setWidget(widget); self.addDockWidget(area, dock)

    def _build_tuning_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        runtime = QWidget(); rlayout = QVBoxLayout(runtime)
        self.runtime_table = QTableWidget(0, 2); self.runtime_table.setHorizontalHeaderLabels(["Runtime tuning key", "Value"])
        self.apply_tuning = QPushButton("Apply selected runtime tuning")
        self.apply_tuning.clicked.connect(self._apply_selected_tuning)
        rlayout.addWidget(self.runtime_table); rlayout.addWidget(self.apply_tuning)
        ddna = QWidget(); dlayout = QVBoxLayout(ddna)
        dlayout.addWidget(QLabel("DDNA is inspect-only in v1. Live genome editing stays locked until the DDNA Viability Validator exists."))
        self.ddna_table = QTableWidget(0, 2); self.ddna_table.setHorizontalHeaderLabels(["DDNA path", "Value"]); dlayout.addWidget(self.ddna_table)
        catalog = QWidget(); clayout = QVBoxLayout(catalog)
        self.catalog_table = QTableWidget(0, 4); self.catalog_table.setHorizontalHeaderLabels(["Class", "Module", "Name", "Default / law"]); clayout.addWidget(self.catalog_table)
        tabs.addTab(runtime, "Runtime knobs"); tabs.addTab(ddna, "DDNA 🔒"); tabs.addTab(catalog, "Defaults / laws")
        return tabs

    def _load_config_catalog(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entries = scan_repo(root)
        self.catalog_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = ["TUNE" if entry.category == "tune" else "LAW 🔒", entry.module, entry.name, entry.value]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value)); item.setFlags(item.flags() & ~Qt.ItemIsEditable); self.catalog_table.setItem(row, col, item)

    def update_snapshot(self, payload: Mapping[str, Any]) -> None:
        self.organs.setPlainText(_pretty({k: payload.get(k) for k in ("queue_depth", "dashboard_queue_depth", "dashboard_dropped_events", "neurons", "bus", "organs", "hypothesis_tuning", "release_tuning")}))
        tunables = payload.get("runtime_tunables", {}) if isinstance(payload.get("runtime_tunables"), Mapping) else {}
        self.runtime_table.setRowCount(len(tunables))
        for row, (key, value) in enumerate(sorted(tunables.items())):
            k = QTableWidgetItem(str(key)); k.setFlags(k.flags() & ~Qt.ItemIsEditable)
            self.runtime_table.setItem(row, 0, k); self.runtime_table.setItem(row, 1, QTableWidgetItem(str(value)))
        ddna = {}
        for prefix in ("profile", "effective"):
            source = payload.get("ddna" if prefix == "profile" else "ddna_effective", {})
            if isinstance(source, Mapping):
                ddna.update({f"{prefix}.{k}": v for k, v in flatten_mapping(source).items()})
        self.ddna_table.setRowCount(len(ddna))
        for row, (key, value) in enumerate(sorted(ddna.items())):
            for col, text in enumerate((key, value)):
                item = QTableWidgetItem(str(text)); item.setFlags(item.flags() & ~Qt.ItemIsEditable); self.ddna_table.setItem(row, col, item)

    def _apply_selected_tuning(self) -> None:
        row = self.runtime_table.currentRow()
        if row < 0:
            return
        key_item, value_item = self.runtime_table.item(row, 0), self.runtime_table.item(row, 1)
        if key_item and value_item:
            asyncio.create_task(self.controller.apply_runtime_tuning(key_item.text(), value_item.text()))

    def add_message(self, msg: UIMessage) -> None:
        raw = {"topic": msg.topic, "source": msg.source, "correlation_id": msg.correlation_id, "payload": safe_json(msg.payload), "meta": safe_json(msg.meta or {})}
        self.raw.appendPlainText(json.dumps(raw, ensure_ascii=False, sort_keys=True))
        corr = msg.correlation_id or "uncorrelated"
        group = self._groups.get(corr)
        if group is None:
            group = QTreeWidgetItem([f"trace {corr[:12]}", "", ""]); self.trace.insertTopLevelItem(0, group); self._groups[corr] = group
            while self.trace.topLevelItemCount() > MAX_TRACE_GROUPS:
                old = self.trace.takeTopLevelItem(self.trace.topLevelItemCount() - 1)
                if old:
                    old_corr = old.text(0).replace("trace ", "");
                    for key in list(self._groups):
                        if key.startswith(old_corr): self._groups.pop(key, None)
        item = QTreeWidgetItem([f"{_stage(msg.topic)} — {msg.topic}", msg.source, _summary(msg.payload)])
        group.addChild(item); group.setExpanded(True); self._messages[id(item)] = msg
        for ref in extract_evidence_refs(msg.payload):
            ev = QTreeWidgetItem([ref["kind"], ref["ref"], msg.topic]); ev.setData(0, Qt.UserRole, ref); self.evidence.insertTopLevelItem(0, ev)
        while self.evidence.topLevelItemCount() > MAX_EVIDENCE_ROWS:
            self.evidence.takeTopLevelItem(self.evidence.topLevelItemCount() - 1)

    def _trace_selected(self) -> None:
        items = self.trace.selectedItems()
        if not items:
            return
        msg = self._messages.get(id(items[0]))
        if msg:
            self.detail.setPlainText(_pretty({"topic": msg.topic, "source": msg.source, "correlation_id": msg.correlation_id, "timestamp": msg.timestamp, "payload": msg.payload, "meta": msg.meta or {}}))

    def _open_evidence(self, item: QTreeWidgetItem, column: int) -> None:
        ref = item.data(0, Qt.UserRole)
        if not isinstance(ref, Mapping) or ref.get("kind") != "file":
            return
        path = Path(str(ref.get("ref") or ""))
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))


class DashboardController:
    def __init__(self, bridge, *, memdir: str | None = None) -> None:
        self.bridge = bridge
        self.assistant_label, self.user_label = load_display_labels(memdir)
        self.transcript = TranscriptWriter(memdir, prefix="dashboard")
        self.presence = PresenceWindow(self)
        self.engineering = EngineeringWindow(self)
        self.timer = QTimer(); self.timer.timeout.connect(self._poll); self.timer.start(UI_POLL_MS)
        self.presence.append_conversation(f"{self.assistant_label} dashboard online. (/quit to close)")
        self.transcript.append_conversation(f"{self.assistant_label} dashboard online. (/quit to close)")

    def submit_text(self, text: str) -> None:
        if text.lower().startswith("/user "):
            name = text.split(" ", 1)[1].strip().strip('"').strip("'")
            self.user_label = "you" if name.lower() in {"clear", "reset", "none", "off"} else (name or self.user_label)
        line = f"{self.user_label}> {text}"; self.presence.append_conversation(line); self.transcript.append_conversation(line)
        local = UIMessage("ui/input_submitted", {"text": text}, "dashboard", {"source": "ui", "channel": "dashboard", "local_echo": True}, timestamp=time.time())
        self.transcript.append_raw(local); self.engineering.add_message(local)
        if text.lower() in {"/quit", "/exit"}:
            QApplication.instance().quit(); return
        asyncio.create_task(self.bridge.send_text(text))

    async def apply_runtime_tuning(self, key: str, value: str) -> None:
        try:
            record = await self.bridge.set_runtime_tuning(key, value)
            self.presence.append_conversation(f"status> tuning {record['key']}: {record['old']} → {record['new']}")
        except Exception as exc:
            self.presence.append_conversation(f"error> tuning change rejected: {exc}")

    def _conversation_line(self, msg: UIMessage, text: str, channel: str, source: str) -> str | None:
        if msg.topic in {"ui/error", "control/error"}:
            return f"error> {text or _summary(msg.payload, 240)}"
        if msg.topic in {"ui/status", "control/status"}:
            return f"status> {text or _summary(msg.payload, 240)}"
        if msg.topic == "act/speech" and channel != "thought":
            return f"{self.assistant_label}> {text}"
        return None

    def _process_line(self, msg: UIMessage, text: str, channel: str, source: str) -> str | None:
        if msg.topic in {"ui/pressure_state", "ui/dashboard_snapshot"}:
            return None
        if msg.topic == "ui/input_submitted":
            return None
        if msg.topic.startswith("control/"):
            return None
        if msg.topic == "act/speech" and channel != "thought":
            return None
        if not text:
            text = _summary(msg.payload, 240)
        if not text:
            return None
        if msg.topic == "act/speech" and channel == "thought":
            return f"thought/probe> {text}"
        return f"{msg.topic}> {text}"

    def _poll(self) -> None:
        for msg in self.bridge.drain_nowait(limit=MAX_MESSAGES_PER_POLL):
            self.transcript.append_raw(msg)
            if msg.topic == "ui/pressure_state" and isinstance(msg.payload, Mapping):
                self.presence.update_pressure(msg.payload); continue
            if msg.topic == "ui/dashboard_snapshot" and isinstance(msg.payload, Mapping):
                self.engineering.update_snapshot(msg.payload); continue
            self.presence.process(msg); self.engineering.add_message(msg)
            text, channel, source = extract_text_and_channels(msg)
            convo_line = self._conversation_line(msg, text, channel, source)
            if convo_line and should_show_in_conversation(msg, text):
                self.presence.append_conversation(convo_line)
                self.transcript.append_conversation(convo_line)
            process_line = self._process_line(msg, text, channel, source)
            if process_line:
                self.presence.append_process(process_line)


def place_windows(presence: QMainWindow, engineering: QMainWindow) -> None:
    screens = QApplication.screens()
    if not screens:
        presence.resize(1000, 800); engineering.resize(1000, 800); return
    if len(screens) >= 2:
        for window, screen in ((presence, screens[0]), (engineering, screens[1])):
            g = screen.availableGeometry(); window.setGeometry(g.x() + DEFAULT_SCREEN_MARGIN_PX, g.y() + DEFAULT_SCREEN_MARGIN_PX, g.width() - 2 * DEFAULT_SCREEN_MARGIN_PX, g.height() - 2 * DEFAULT_SCREEN_MARGIN_PX)
    else:
        g = screens[0].availableGeometry(); half = max(600, g.width() // 2)
        presence.setGeometry(g.x(), g.y(), half, g.height()); engineering.setGeometry(g.x() + half, g.y(), g.width() - half, g.height())


def create_dashboard(bridge, *, memdir: str | None = None) -> DashboardController:
    controller = DashboardController(bridge, memdir=memdir)
    place_windows(controller.presence, controller.engineering)
    controller.presence.show(); controller.engineering.show()
    return controller
