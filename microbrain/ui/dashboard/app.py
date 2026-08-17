"""Two-window PySide6 engineering dashboard for MicroBrain."""

from __future__ import annotations

import asyncio
import html
import json
import time
from pathlib import Path
from typing import Any, Mapping

from PySide6.QtCore import QPointF, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QToolButton,
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
from microbrain.vision_state import (
    bbox_xywh,
    has_visual_motion_salience,
    visual_object_uncertain,
    visual_ref_text,
)
from microbrain.ui.dashboard.status_signals import (
    capability_counts,
    capability_short_label,
    capability_signal_map,
)

# ---------------------------------------------------------------------------
# Behavioral tuning
# ---------------------------------------------------------------------------

UI_POLL_MS = 50
MAX_MESSAGES_PER_POLL = 120
MAX_RAW_LINES = 1500
MAX_TRACE_GROUPS = 120
MAX_EVIDENCE_ROWS = 400
VISION_OVERLAY_LINE_WIDTH = 2
VISION_OVERLAY_SELECTED_LINE_WIDTH = 4
VISION_SELECTION_FLASH_MS = 850
VISION_MOTION_HIGHLIGHT_MS = 1200
VISION_CONFIDENT_THRESHOLD = 0.75
# Attentional overlay colors: focus, noticed/peripheral, recent motion.
VISION_COLOR_FOCUSED = "#35c759"
VISION_COLOR_NOTICED = "#0a84ff"
VISION_COLOR_MOTION = "#ffd60a"
VISION_COLOR_HAZARD = "#ff453a"
VISION_COLOR_LOST = "#8e8e93"
VISION_COLOR_SELECTED = "#ffffff"
# Backward-compatible aliases for older dashboard/tests/imports.
VISION_COLOR_IDENTIFIED = VISION_COLOR_FOCUSED
VISION_COLOR_UNCERTAIN = VISION_COLOR_MOTION
VISION_COLOR_UNKNOWN = VISION_COLOR_NOTICED
LOG_TAIL_POLL_MS = 500
LOG_TAIL_INITIAL_BYTES = 65536
DEFAULT_SCREEN_MARGIN_PX = 20
COMPACT_PANEL_HEIGHT_PX = 34
COMPACT_PANEL_WIDTH_PX = 190
COMPACT_DOCK_HEIGHT_PX = 36
COMPACT_DOCK_WIDTH_PX = 210
MAX_SLEARN_LOG_LINES = 400
CAPABILITY_GOOD_COLOR = "#35c759"
CAPABILITY_BAD_COLOR = "#ff453a"
CAPABILITY_UNKNOWN_COLOR = "#8e8e93"
HEARTBEAT_GOOD_COLOR = "#35c759"
HEARTBEAT_ALERT_COLOR = "#ffd60a"
HEARTBEAT_BAD_COLOR = "#ff453a"
HEARTBEAT_UNKNOWN_COLOR = "#8e8e93"

# ---------------------------------------------------------------------------
# Required static constants
# ---------------------------------------------------------------------------

WORKSPACE_SETTINGS_ORG = "ExisMechanica"
WORKSPACE_SETTINGS_APP = "MicroBrainDashboard"
WORKSPACE_SETTINGS_VERSION = 2

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


def _settings_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _settings_sizes(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            try:
                out.append(int(item))
            except Exception:
                pass
        return out
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    out = []
    for item in text.split(","):
        try:
            out.append(int(item.strip()))
        except Exception:
            pass
    return out


def _sizes_text(splitter: QSplitter) -> str:
    return ",".join(str(int(v)) for v in splitter.sizes())


def _is_slearn_diagnostic(msg: UIMessage) -> bool:
    kind = str((msg.meta or {}).get("kind") or "").lower()
    return msg.topic.startswith("slearn/") or kind.startswith("slearn_")


def _is_status_instrument_event(msg: UIMessage) -> bool:
    """Telemetry rendered as an instrument, not as a scrolling trace line."""

    return msg.topic in {"capability/state", "ui/vision_current"}


def _is_ephemeral_visual_sample(msg: UIMessage) -> bool:
    """High-rate vision samples that must not become dashboard disk/history sludge."""

    return msg.topic in {"percept/vision", "percept/vision/features", "ui/vision_current"}


class CapabilityStatusStrip(QWidget):
    """Compact red/green capability lamps for Presence and Engineering."""

    def __init__(self, title: str = "CAP") -> None:
        super().__init__()
        self._title = title
        self._payload: dict[str, Any] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)
        self.label = QLabel(f"{html.escape(title)} — waiting")
        self.label.setTextFormat(Qt.RichText)
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.label.setToolTip("Capability / available-signal status. Green = available, red = unavailable.")
        layout.addWidget(self.label, 1)

    def update_payload(self, payload: Mapping[str, Any] | None) -> None:
        self._payload = dict(payload) if isinstance(payload, Mapping) else {}
        signals = capability_signal_map(self._payload)
        if not signals:
            self.label.setText(f'{html.escape(self._title)} <span style="color:{CAPABILITY_UNKNOWN_COLOR}">●</span> waiting')
            return
        up, total = capability_counts(self._payload)
        bits = [f"<b>{html.escape(self._title)}</b> {up}/{total}"]
        for name, available in signals.items():
            color = CAPABILITY_GOOD_COLOR if available else CAPABILITY_BAD_COLOR
            label = html.escape(capability_short_label(name))
            bits.append(f'<span style="color:{color}">●</span>&nbsp;{label}')
        self.label.setText("&nbsp;&nbsp;".join(bits))

    def compact_text(self) -> str:
        up, total = capability_counts(self._payload)
        return f"cap {up}/{total}" if total else "cap waiting"


class HeartbeatStatusStrip(QWidget):
    """Compact body-clock health / arousal instrument for Engineering."""

    def __init__(self) -> None:
        super().__init__()
        self._heartbeat: dict[str, Any] = {}
        self._adrenaline: dict[str, Any] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)
        self.label = QLabel('HEART <span style="color:#8e8e93">●</span> waiting')
        self.label.setTextFormat(Qt.RichText)
        self.label.setToolTip(
            "Canonical body heartbeat. 20 TPS is scheduling; monotonic timestamps are elapsed-time truth."
        )
        layout.addWidget(self.label, 1)

    def update_payloads(
        self,
        heartbeat: Mapping[str, Any] | None,
        adrenaline: Mapping[str, Any] | None,
    ) -> None:
        self._heartbeat = dict(heartbeat) if isinstance(heartbeat, Mapping) else {}
        self._adrenaline = dict(adrenaline) if isinstance(adrenaline, Mapping) else {}

        last_ts = float(self._heartbeat.get("last_epoch_s", 0.0) or 0.0)
        age = max(0.0, time.time() - last_ts) if last_ts else 9999.0
        hz = float(self._heartbeat.get("actual_hz_ema", 0.0) or 0.0)
        jitter = float(self._heartbeat.get("jitter_ms_ema", 0.0) or 0.0)
        missed = int(self._heartbeat.get("missed_total", 0) or 0)
        mode = str(self._adrenaline.get("mode", "normal") or "normal").upper()

        if not last_ts:
            lamp = HEARTBEAT_UNKNOWN_COLOR
            status = "waiting"
        elif age > 0.50:
            lamp = HEARTBEAT_BAD_COLOR
            status = f"stale {age:.1f}s"
        elif mode == "EMERGENCY":
            lamp = HEARTBEAT_BAD_COLOR
            status = "EMERGENCY"
        elif mode == "ALERT":
            lamp = HEARTBEAT_ALERT_COLOR
            status = "ALERT"
        else:
            lamp = HEARTBEAT_GOOD_COLOR
            status = "NORMAL"

        hz_text = f"{hz:.1f} TPS" if hz > 0.0 else "-- TPS"
        self.label.setText(
            f'<b>HEART</b> <span style="color:{lamp}">●</span>&nbsp;{hz_text}'
            f'&nbsp;&nbsp; jitter {jitter:.1f} ms'
            f'&nbsp;&nbsp; missed {missed}'
            f'&nbsp;&nbsp; {html.escape(status)}'
        )


class CompactablePanel(QFrame):
    """A splitter-friendly panel that can collapse without changing its slot.

    In a horizontal splitter it becomes a narrow instrument strip. In a vertical
    splitter it becomes a short header strip. Expanding restores the previous
    splitter allocation when possible.
    """

    def __init__(self, title: str, content: QWidget) -> None:
        super().__init__()
        self.title = title
        self.content = content
        self._compact = False
        self._saved_splitter_sizes: list[int] = []
        self.setFrameShape(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 2, 0)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600;")
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.compact_button = QToolButton()
        self.compact_button.setText("−")
        self.compact_button.setToolTip("Compact this panel in place")
        self.compact_button.clicked.connect(self.toggle_compact)
        header.addWidget(self.title_label)
        header.addWidget(self.status_label, 1)
        header.addWidget(self.compact_button)
        layout.addLayout(header)
        layout.addWidget(content, 1)

    @property
    def is_compact(self) -> bool:
        return self._compact

    def set_status(self, text: str) -> None:
        self.status_label.setText(str(text or ""))

    def toggle_compact(self) -> None:
        self.set_compact(not self._compact)

    def set_compact(self, compact: bool) -> None:
        compact = bool(compact)
        if compact == self._compact:
            return
        splitter = self.parentWidget() if isinstance(self.parentWidget(), QSplitter) else None
        index = splitter.indexOf(self) if splitter is not None else -1
        if compact:
            if splitter is not None:
                self._saved_splitter_sizes = list(splitter.sizes())
            self.content.setVisible(False)
            self.compact_button.setText("+")
            self.compact_button.setToolTip("Expand this panel")
            self._compact = True
            self._apply_compact_constraints(splitter)
            if splitter is not None and index >= 0:
                sizes = list(self._saved_splitter_sizes)
                if len(sizes) == splitter.count():
                    compact_size = (
                        COMPACT_PANEL_WIDTH_PX
                        if splitter.orientation() == Qt.Horizontal
                        else COMPACT_PANEL_HEIGHT_PX
                    )
                    freed = max(0, sizes[index] - compact_size)
                    sizes[index] = compact_size
                    others = [i for i in range(len(sizes)) if i != index]
                    if others and freed:
                        target = max(others, key=lambda i: sizes[i])
                        sizes[target] += freed
                    splitter.setSizes(sizes)
            return

        self._compact = False
        self.compact_button.setText("−")
        self.compact_button.setToolTip("Compact this panel in place")
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self.content.setVisible(True)
        if splitter is not None and len(self._saved_splitter_sizes) == splitter.count():
            splitter.setSizes(self._saved_splitter_sizes)

    def _apply_compact_constraints(self, splitter: QSplitter | None) -> None:
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        if splitter is not None and splitter.orientation() == Qt.Horizontal:
            self.setMinimumWidth(COMPACT_PANEL_WIDTH_PX)
            self.setMaximumWidth(COMPACT_PANEL_WIDTH_PX)
        else:
            self.setMinimumHeight(COMPACT_PANEL_HEIGHT_PX)
            self.setMaximumHeight(COMPACT_PANEL_HEIGHT_PX)


class CompactDockWidget(QDockWidget):
    """QDockWidget with a compact-in-place control while retaining Qt docking."""

    def __init__(self, title: str, widget: QWidget, parent: QMainWindow) -> None:
        super().__init__(title, parent)
        self.base_title = title
        self._compact_status = ""
        self._compact = False
        self._saved_min = (self.minimumWidth(), self.minimumHeight())
        self._saved_max = (self.maximumWidth(), self.maximumHeight())
        self.setWidget(widget)
        self.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable)
        self.compact_button = QToolButton(self)
        self.compact_button.setText("−")
        self.compact_button.setFixedSize(18, 18)
        self.compact_button.setToolTip("Compact this dock in place")
        self.compact_button.clicked.connect(self.toggle_compact)
        self.dockLocationChanged.connect(lambda _area: self._reapply_compact_constraints())
        self.topLevelChanged.connect(lambda _floating: self._reapply_compact_constraints())

    @property
    def is_compact(self) -> bool:
        return self._compact

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Leave room for Qt's native float/close buttons on the far right.
        self.compact_button.move(max(4, self.width() - 76), 3)
        self.compact_button.raise_()

    def set_compact_status(self, text: str) -> None:
        self._compact_status = " ".join(str(text or "").split())
        self._refresh_title()

    def toggle_compact(self) -> None:
        self.set_compact(not self._compact)

    def set_compact(self, compact: bool) -> None:
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact
        content = self.widget()
        if content is not None:
            content.setVisible(not compact)
        self.compact_button.setText("+" if compact else "−")
        self.compact_button.setToolTip("Expand this dock" if compact else "Compact this dock in place")
        if compact:
            self._reapply_compact_constraints()
        else:
            self.setMinimumSize(*self._saved_min)
            self.setMaximumSize(*self._saved_max)
        self._refresh_title()

    def _refresh_title(self) -> None:
        if self._compact and self._compact_status:
            self.setWindowTitle(f"{self.base_title} — {self._compact_status}")
        else:
            self.setWindowTitle(self.base_title)

    def _reapply_compact_constraints(self) -> None:
        if not self._compact:
            return
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        parent = self.parentWidget()
        area = parent.dockWidgetArea(self) if isinstance(parent, QMainWindow) else Qt.NoDockWidgetArea
        if self.isFloating() or area in {Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea, Qt.NoDockWidgetArea}:
            self.setMinimumHeight(COMPACT_DOCK_HEIGHT_PX)
            self.setMaximumHeight(COMPACT_DOCK_HEIGHT_PX)
        else:
            self.setMinimumWidth(COMPACT_DOCK_WIDTH_PX)
            self.setMaximumWidth(COMPACT_DOCK_WIDTH_PX)


class SlearnJobWidget(QWidget):
    """Engineering-only SLEARN workbench/status panel.

    It accepts today's chunk-status payloads and the richer preflight/bucket/
    workspace/completion payloads planned for the ingestion rewrite.
    """

    def __init__(self) -> None:
        super().__init__()
        self._status_callback = None
        self._last_file = ""
        self._last_mode = ""
        self._last_status = "idle"
        self._last_history_signature = ""
        self._last_composer_health: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.state_value = QLabel("idle")
        self.file_value = QLabel("—")
        self.mode_value = QLabel("—")
        self.counts_value = QLabel("files 0 | staged 0 | applied 0")
        self.workspace_value = QLabel("—")
        self.composer_value = QLabel("idle")
        self.composer_worker_value = QLabel("—")
        self.composer_queue_value = QLabel("—")
        self.composer_cycle_value = QLabel("—")
        self.composer_fault_value = QLabel("none")
        form.addRow("State", self.state_value)
        form.addRow("File", self.file_value)
        form.addRow("Mode", self.mode_value)
        form.addRow("Counts", self.counts_value)
        form.addRow("Workspace", self.workspace_value)
        form.addRow("Composer", self.composer_value)
        form.addRow("Worker", self.composer_worker_value)
        form.addRow("Queue", self.composer_queue_value)
        form.addRow("Cycle", self.composer_cycle_value)
        form.addRow("Fault", self.composer_fault_value)
        layout.addLayout(form)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("idle")
        layout.addWidget(self.progress)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setMaximumBlockCount(MAX_SLEARN_LOG_LINES)
        self.history.setPlaceholderText("SLEARN preflight / ingest / cleanup / completion feedback")
        layout.addWidget(self.history, 1)

    def set_status_callback(self, callback) -> None:
        self._status_callback = callback
        self._emit_compact_status()

    def update_snapshot(self, payload: Mapping[str, Any]) -> None:
        merged: dict[str, Any] = dict(payload)
        last_result = payload.get("last_result")
        if isinstance(last_result, Mapping):
            for key, value in last_result.items():
                merged.setdefault(key, value)
        self._apply_payload(merged, topic="snapshot", append_history=False)

    def update_event(self, msg: UIMessage) -> None:
        payload = dict(msg.payload) if isinstance(msg.payload, Mapping) else {"summary": _summary(msg.payload)}
        self._apply_payload(payload, topic=msg.topic, append_history=True)

    def _apply_payload(self, payload: Mapping[str, Any], *, topic: str, append_history: bool) -> None:
        file_value = (
            payload.get("active_file")
            or payload.get("file")
            or payload.get("completed_file")
            or payload.get("source_name")
            or ""
        )
        if file_value:
            try:
                self._last_file = Path(str(file_value)).name
            except Exception:
                self._last_file = str(file_value)

        mode = payload.get("mode") or payload.get("ingest_mode") or payload.get("selected_mode") or ""
        if mode:
            self._last_mode = str(mode).upper()

        status = str(payload.get("status") or "").strip().lower()
        summary = str(payload.get("summary") or payload.get("reason") or "").strip()
        if topic.startswith("learning/"):
            status = topic.split("/", 1)[1]
        elif payload.get("completed_file") or "completed" in summary.lower():
            status = "completed"
        elif status == "" and (self._last_file or payload.get("active_file")):
            status = "active"
        if status:
            self._last_status = status

        files_done = int(payload.get("files_completed_count", 0) or 0)
        applied = int(payload.get("rules_applied_total", payload.get("applied", 0)) or 0)
        accepted = payload.get("accepted")
        duplicates = payload.get("duplicates")
        rejected = payload.get("rejected")
        outstanding = int(payload.get("outstanding_batches", 0) or 0)
        counter_parts = [f"files {files_done}"]
        if accepted is not None:
            counter_parts.append(f"accepted {int(accepted or 0)}")
        counter_parts.append(f"pending {outstanding}")
        if duplicates is not None:
            counter_parts.append(f"dupes {int(duplicates or 0)}")
        if rejected is not None:
            counter_parts.append(f"rejected {int(rejected or 0)}")
        if applied:
            counter_parts.append(f"committed total {applied}")

        workspace_bits: list[str] = []
        workspace = payload.get("workspace")
        if isinstance(workspace, Mapping):
            if "clean" in workspace:
                workspace_bits.append("clean" if workspace.get("clean") else "dirty")
            if "baseline_restored" in workspace:
                workspace_bits.append("baseline restored" if workspace.get("baseline_restored") else "baseline pending")
        if "workspace_clean" in payload:
            workspace_bits.append("clean" if payload.get("workspace_clean") else "dirty")
        if "baseline_restored" in payload:
            workspace_bits.append("baseline restored" if payload.get("baseline_restored") else "baseline pending")
        warnings = payload.get("warnings")
        if isinstance(warnings, (list, tuple)) and warnings:
            workspace_bits.append(f"warnings {len(warnings)}")

        composer_busy = bool(payload.get("composer_busy", False))
        composer_deferred = bool(payload.get("composer_learned_deferred", False))
        flush_batches = int(payload.get("composer_flush_batches", 0) or 0)
        if composer_busy:
            composer_text = "committing learned memory"
        elif composer_deferred:
            composer_text = f"buffering {outstanding}/{flush_batches or '?'} batches"
        elif outstanding:
            composer_text = f"queued {outstanding} batch(es)"
        else:
            composer_text = "idle / caught up"

        self.state_value.setText(self._last_status or "idle")
        self.file_value.setText(self._last_file or "—")
        self.mode_value.setText(self._last_mode or "—")
        self.counts_value.setText(" | ".join(counter_parts))
        self.workspace_value.setText(" | ".join(workspace_bits) if workspace_bits else "—")
        self.composer_value.setText(composer_text)
        self._update_composer_health(payload)
        self._update_progress(payload)

        if append_history:
            stamp = time.strftime("%H:%M:%S")
            text = summary or _summary(payload, 260)
            signature = f"{topic}|{text}"
            if text and signature != self._last_history_signature:
                self.history.appendPlainText(f"{stamp}  {topic}> {text}")
                self._last_history_signature = signature
        self._emit_compact_status()


    @staticmethod
    def _age_text(ts: object) -> str:
        try:
            stamp = float(ts or 0.0)
        except (TypeError, ValueError):
            return "never"
        if stamp <= 0.0:
            return "never"
        age = max(0.0, time.time() - stamp)
        if age < 10.0:
            return f"{age:.1f}s ago"
        if age < 120.0:
            return f"{int(age)}s ago"
        if age < 7200.0:
            return f"{age / 60.0:.1f}m ago"
        return f"{age / 3600.0:.1f}h ago"

    @staticmethod
    def _tier_counts_text(value: object) -> str:
        if not isinstance(value, Mapping):
            return "0"
        bits = []
        labels = (("learned", "L"), ("now", "N"), ("short", "S"), ("long", "G"))
        for key, label in labels:
            count = int(value.get(key, 0) or 0)
            if count:
                bits.append(f"{label}:{count}")
        return " ".join(bits) if bits else "0"

    def _update_composer_health(self, payload: Mapping[str, Any]) -> None:
        health = payload.get("composer_health")
        if isinstance(health, Mapping) and health:
            self._last_composer_health = dict(health)
        elif self._last_composer_health:
            # SLEARN progress/status events often do not carry the full composer
            # health snapshot.  Keep the last real health instead of flickering
            # back to legacy mode between bus messages.
            health = dict(self._last_composer_health)
        else:
            self.composer_worker_value.setText("legacy telemetry only")
            pending = payload.get("composer_pending", {})
            self.composer_queue_value.setText(f"pending {self._tier_counts_text(pending)}")
            self.composer_cycle_value.setText("—")
            self.composer_fault_value.setText(str(payload.get("composer_last_error", "") or "none"))
            return

        state = str(health.get("state", "unknown") or "unknown")
        task_alive = bool(health.get("task_alive", False))
        busy_age = float(health.get("busy_age_s", 0.0) or 0.0)
        pulse_age = self._age_text(health.get("ts"))
        phase = health.get("compose_phase")
        if not isinstance(phase, Mapping):
            phase = {}
        phase_name = str(phase.get("phase", "") or "")
        phase_tier = str(phase.get("tier", "") or "")
        phase_detail = str(phase.get("detail", "") or "")
        phase_file = str(phase.get("file", "") or "")
        phase_age = float(phase.get("phase_age_s", 0.0) or 0.0)
        phase_pulse_age = float(phase.get("phase_pulse_age_s", 0.0) or 0.0)
        worker_bits = ["alive" if task_alive else "DEAD", state, f"pulse {pulse_age}"]
        if busy_age > 0.0:
            worker_bits.append(f"busy {busy_age:.1f}s")
        if phase_name and phase_name != "idle":
            phase_bits = [phase_name]
            if phase_tier:
                phase_bits.append(phase_tier)
            if phase_age > 0.0:
                phase_bits.append(f"{phase_age:.1f}s")
            if phase_pulse_age > 5.0:
                phase_bits.append(f"pulse stale {phase_pulse_age:.1f}s")
            worker_bits.append("phase " + " ".join(phase_bits))
        if health.get("lock_exists"):
            worker_bits.append(f"lock {float(health.get('lock_age_s', 0.0) or 0.0):.1f}s")
        queue_scan_age = float(health.get("queue_scan_age_s", 0.0) or 0.0)
        if health.get("queue_scan_stalled"):
            worker_bits.append(f"queue scan STALLED {queue_scan_age:.1f}s")
        elif health.get("queue_scan_running"):
            worker_bits.append(f"queue scan {queue_scan_age:.1f}s")
        scan_tiers = health.get("scan_tiers")
        if isinstance(scan_tiers, (list, tuple)) and scan_tiers:
            worker_bits.append("scan " + ",".join(str(t) for t in scan_tiers))
        reason = str(health.get("target_tiers_reason", "") or "")
        if reason and reason != "normal":
            worker_bits.append(reason)
        self.composer_worker_value.setText(" | ".join(worker_bits))

        pending_text = self._tier_counts_text(health.get("pending", {}))
        processing_text = self._tier_counts_text(health.get("processing", {}))
        self.composer_queue_value.setText(f"pending {pending_text} | processing {processing_text}")

        last_status = health.get("last_status")
        if not isinstance(last_status, Mapping):
            last_status = {}
        cycle_bits = [f"#{int(health.get('cycle_index', 0) or 0)}"]
        elapsed = float(health.get("last_cycle_elapsed_s", 0.0) or 0.0)
        if elapsed > 0.0:
            cycle_bits.append(f"last {elapsed:.2f}s")
        if health.get("last_success_ts"):
            cycle_bits.append(f"good {self._age_text(health.get('last_success_ts'))}")
        files_processed = int(last_status.get("files_processed", 0) or 0)
        rows_applied = int(last_status.get("rows_applied", 0) or 0)
        if files_processed or rows_applied:
            cycle_bits.append(f"{files_processed} files / {rows_applied} rows")
        if phase_name and phase_name != "idle":
            phase_summary = phase_name
            if phase_tier:
                phase_summary += f" {phase_tier}"
            if phase_detail:
                phase_summary += f" · {phase_detail}"
            if phase_file:
                phase_summary += f" · {phase_file[:42]}"
            ops_applied = int(phase.get("operations_applied", 0) or 0)
            ops_loaded = int(phase.get("operations_loaded", 0) or 0)
            if ops_loaded:
                phase_summary += f" · ops {ops_applied}/{ops_loaded}"
            cycle_bits.append(phase_summary)
        self.composer_cycle_value.setText(" | ".join(cycle_bits))

        error = str(health.get("last_error", "") or "")
        error_type = str(health.get("last_error_type", "") or "")
        queue_scan_error = str(health.get("queue_scan_error", "") or "")
        if error:
            fault = f"{error_type + ': ' if error_type else ''}{error}"
            if health.get("last_error_ts"):
                fault += f" | {self._age_text(health.get('last_error_ts'))}"
            self.composer_fault_value.setText(fault)
        elif state == "busy_long" and phase_name:
            phase_fault = f"composer phase long: {phase_name}"
            if phase_tier:
                phase_fault += f"/{phase_tier}"
            if phase_age:
                phase_fault += f" {phase_age:.1f}s"
            if phase_detail:
                phase_fault += f" | {phase_detail}"
            if phase_file:
                phase_fault += f" | {phase_file}"
            self.composer_fault_value.setText(phase_fault)
        elif queue_scan_error:
            self.composer_fault_value.setText(f"queue scan: {queue_scan_error}")
        elif health.get("queue_scan_stalled"):
            self.composer_fault_value.setText("queue directory scan stalled; composer/UI remain alive")
        else:
            self.composer_fault_value.setText("none")

        if state in {"worker_dead", "error"}:
            self.composer_worker_value.setStyleSheet(f"color: {HEARTBEAT_BAD_COLOR};")
        elif state == "busy_long" or health.get("queue_scan_stalled"):
            self.composer_worker_value.setStyleSheet(f"color: {HEARTBEAT_ALERT_COLOR};")
        else:
            self.composer_worker_value.setStyleSheet("")

    def _update_progress(self, payload: Mapping[str, Any]) -> None:
        progress_value = payload.get("progress_pct", payload.get("progress"))
        percent: int | None = None
        if isinstance(progress_value, (int, float)):
            raw = float(progress_value)
            if 0.0 <= raw <= 1.0:
                raw *= 100.0
            percent = max(0, min(100, int(round(raw))))
        else:
            processed = payload.get("processed")
            total = payload.get("total") or payload.get("total_lines")
            if isinstance(processed, (int, float)) and isinstance(total, (int, float)) and float(total) > 0:
                percent = max(0, min(100, int(round((float(processed) / float(total)) * 100.0))))

        if self._last_status == "completed":
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("completed")
        elif percent is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(percent)
            self.progress.setFormat(f"{percent}%")
        elif self._last_status in {"active", "running", "ingesting", "preflight", "cleaning"}:
            self.progress.setRange(0, 0)
            self.progress.setFormat(self._last_status)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setFormat(self._last_status or "idle")

    def _emit_compact_status(self) -> None:
        if not callable(self._status_callback):
            return
        bits = [self._last_status]
        if self._last_mode:
            bits.append(self._last_mode)
        if self._last_file:
            bits.append(self._last_file)
        self._status_callback(" · ".join(bit for bit in bits if bit))


class VisionCanvas(QWidget):
    """Live camera canvas with overlays supplied by the vision organ.

    The dashboard does not infer objects. It only renders the current object
    state that the vision system reports.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(480, 300)
        self._pixmap = QPixmap()
        self._overlays: list[dict[str, Any]] = []
        self._source_size = (0, 0)
        self._label = "No vision frame yet"
        self._show_boxes = True
        self._show_labels = True
        self._show_confidence = True
        self._show_track_ids = True
        self._show_motion = False
        self._highlight_track_id = ""
        self._focus_track_id = ""
        self._highlight_generation = 0
        self._motion_seen: dict[str, float] = {}
        self._frame_frozen = False
        self._frozen_label = ""
        self._freeze_generation = 0

    def set_frame(self, path: str, width: int = 0, height: int = 0) -> None:
        if self._frame_frozen:
            return
        candidate = Path(path)
        pixmap = QPixmap(str(candidate)) if candidate.exists() else QPixmap()
        if not pixmap.isNull():
            self._pixmap = pixmap
            self._source_size = (width or pixmap.width(), height or pixmap.height())
            self._label = candidate.name
            self.update()

    def set_frame_bytes(self, data: bytes | bytearray, width: int = 0, height: int = 0, label: str = "RAM frame") -> None:
        if self._frame_frozen:
            return
        pixmap = QPixmap()
        if not isinstance(data, (bytes, bytearray)) or not pixmap.loadFromData(bytes(data)):
            return
        self._pixmap = pixmap
        self._source_size = (width or pixmap.width(), height or pixmap.height())
        self._label = str(label or "RAM frame")
        self.update()

    def set_overlays(self, overlays: list[dict[str, Any]]) -> None:
        if self._frame_frozen:
            return
        self._overlays = self.decorate_objects(overlays[-64:])
        self.update()

    def decorate_objects(self, objects: list[dict[str, Any]] | list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Add UI-only attention markers without changing vision truth.

        Green/blue/yellow are dashboard exposure states, not identity claims:
        focused, noticed/peripheral, and recent motion. Uncertainty stays in
        the reference text as a ``?`` suffix, e.g. ``vobj:07?``.
        """

        now = time.monotonic()
        expire_after = max(0.1, VISION_MOTION_HIGHLIGHT_MS / 1000.0)
        decorated: list[dict[str, Any]] = []
        active_ids: set[str] = set()
        for index, raw in enumerate(objects):
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            track_id = str(item.get("track_id") or item.get("object_id") or item.get("proto_id") or "").strip()
            if track_id:
                active_ids.add(track_id)
                if has_visual_motion_salience(item):
                    self._motion_seen[track_id] = now
            last_motion = self._motion_seen.get(track_id, 0.0) if track_id else 0.0
            motion_recent = bool(track_id and last_motion and (now - last_motion) <= expire_after)
            focused = bool(track_id and track_id == self._focus_track_id)
            if focused:
                attention_state = "focused"
            elif motion_recent:
                attention_state = "motion"
            else:
                attention_state = "noticed"
            item["_ui_attention_state"] = attention_state
            item["_ui_motion_recent"] = motion_recent
            item["_ui_uncertain_ref"] = visual_object_uncertain(item, confidence_threshold=VISION_CONFIDENT_THRESHOLD)
            item["_ui_ref"] = visual_ref_text(item, fallback_index=index, confidence_threshold=VISION_CONFIDENT_THRESHOLD)
            decorated.append(item)
        stale_ids = [track_id for track_id, ts in self._motion_seen.items() if track_id not in active_ids or now - ts > expire_after]
        for track_id in stale_ids:
            self._motion_seen.pop(track_id, None)
        return decorated

    def set_frozen(self, frozen: bool) -> None:
        """Freeze the currently rendered frame/overlays for visual teaching."""
        frozen = bool(frozen)
        if frozen == self._frame_frozen:
            return
        self._frame_frozen = frozen
        self._freeze_generation += 1
        if frozen:
            self._frozen_label = self._label
        else:
            self._frozen_label = ""
        self.update()

    def snapshot_context(self) -> dict[str, Any]:
        return {
            "frame_label": self._frozen_label or self._label,
            "source_width": int(self._source_size[0] or 0),
            "source_height": int(self._source_size[1] or 0),
            "frozen": bool(self._frame_frozen),
            "freeze_generation": int(self._freeze_generation),
        }

    def set_display_options(
        self,
        *,
        boxes: bool | None = None,
        labels: bool | None = None,
        confidence: bool | None = None,
        track_ids: bool | None = None,
        motion: bool | None = None,
    ) -> None:
        if boxes is not None:
            self._show_boxes = bool(boxes)
        if labels is not None:
            self._show_labels = bool(labels)
        if confidence is not None:
            self._show_confidence = bool(confidence)
        if track_ids is not None:
            self._show_track_ids = bool(track_ids)
        if motion is not None:
            self._show_motion = bool(motion)
        self.update()

    def flash_object(self, track_id: str) -> None:
        self._highlight_track_id = str(track_id or "")
        self._focus_track_id = self._highlight_track_id
        self._overlays = self.decorate_objects(self._overlays)
        self._highlight_generation += 1
        generation = self._highlight_generation
        self.update()

        def clear() -> None:
            if generation != self._highlight_generation:
                return
            self._highlight_track_id = ""
            self.update()

        QTimer.singleShot(VISION_SELECTION_FLASH_MS, clear)

    @staticmethod
    def _state_key(item: Mapping[str, Any]) -> str:
        status = str(item.get("status") or "").strip().lower()
        if bool(item.get("hazard", False)) or status in {"hazard", "danger", "emergency"}:
            return "hazard"
        if status in {"lost", "missing", "stale"}:
            return "lost"
        attention_state = str(item.get("_ui_attention_state") or "").strip().lower()
        if attention_state in {"focused", "focus", "selected"}:
            return "focused"
        if attention_state in {"motion", "moving", "changed"} or bool(item.get("_ui_motion_recent", False)):
            return "motion"
        return "noticed"

    @staticmethod
    def _state_color(item: Mapping[str, Any]) -> QColor:
        key = VisionCanvas._state_key(item)
        return QColor(
            {
                "focused": VISION_COLOR_FOCUSED,
                "noticed": VISION_COLOR_NOTICED,
                "motion": VISION_COLOR_MOTION,
                "identified": VISION_COLOR_FOCUSED,
                "uncertain": VISION_COLOR_NOTICED,
                "unknown": VISION_COLOR_NOTICED,
                "hazard": VISION_COLOR_HAZARD,
                "lost": VISION_COLOR_LOST,
            }.get(key, VISION_COLOR_NOTICED)
        )

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
        if self._frame_frozen:
            painter.setPen(QPen(QColor(VISION_COLOR_SELECTED), 2))
            painter.drawText(x0 + 8, y0 + 20, "FROZEN FRAME")
        sw, sh = self._source_size
        if sw <= 0 or sh <= 0:
            return
        sx, sy = scaled.width() / sw, scaled.height() / sh

        for item in self._overlays:
            coords = bbox_xywh(item.get("bbox"), source_width=sw, source_height=sh)
            if coords is None:
                continue
            bx, by, bw, bh = coords
            rx, ry, rw, rh = x0 + bx * sx, y0 + by * sy, bw * sx, bh * sy
            track_id = str(item.get("track_id") or item.get("object_id") or item.get("proto_id") or "")
            flash_selected = bool(track_id and track_id == self._highlight_track_id)
            focused = bool(track_id and track_id == self._focus_track_id)
            color = self._state_color(item)
            width = VISION_OVERLAY_SELECTED_LINE_WIDTH if (focused or flash_selected) else VISION_OVERLAY_LINE_WIDTH
            pen = QPen(color, width)
            if str(item.get("status") or "").lower() in {"lost", "missing", "searching"}:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)

            if self._show_boxes:
                contour = item.get("contour")
                polygon_points: list[QPointF] = []
                if isinstance(contour, (list, tuple)):
                    for point in contour:
                        if not isinstance(point, (list, tuple)) or len(point) < 2:
                            continue
                        try:
                            px, py = float(point[0]), float(point[1])
                        except Exception:
                            continue
                        polygon_points.append(QPointF(x0 + px * sx, y0 + py * sy))
                if len(polygon_points) >= 3:
                    painter.drawPolygon(QPolygonF(polygon_points))
                else:
                    painter.drawRect(int(rx), int(ry), int(rw), int(rh))

            parts: list[str] = []
            if self._show_track_ids and track_id:
                parts.append(str(item.get("_ui_ref") or visual_ref_text(item, confidence_threshold=VISION_CONFIDENT_THRESHOLD))[:24])
            if self._show_labels:
                parts.append(str(item.get("label") or "object"))
            if self._show_confidence and item.get("confidence") is not None:
                try:
                    parts.append(f"{float(item.get('confidence')):.2f}")
                except Exception:
                    pass
            if parts:
                painter.drawText(int(rx) + 3, max(14, int(ry) + 14), "  ".join(parts))

            if self._show_motion:
                motion = item.get("motion")
                dx = dy = 0.0
                if isinstance(motion, Mapping):
                    try:
                        dx = float(motion.get("dx", motion.get("x", 0.0)) or 0.0)
                        dy = float(motion.get("dy", motion.get("y", 0.0)) or 0.0)
                    except Exception:
                        dx = dy = 0.0
                elif isinstance(motion, (list, tuple)) and len(motion) >= 2:
                    try:
                        dx, dy = float(motion[0]), float(motion[1])
                    except Exception:
                        dx = dy = 0.0
                if abs(dx) <= 1.5 and abs(dy) <= 1.5:
                    dx *= sw
                    dy *= sh
                if dx or dy:
                    cx, cy = rx + rw / 2.0, ry + rh / 2.0
                    painter.drawLine(int(cx), int(cy), int(cx + dx * sx), int(cy + dy * sy))

            # Clicking an object in the left inspector produces a short, explicit
            # cue from the camera's left edge to the corresponding object.
            if flash_selected:
                cx, cy = rx + rw / 2.0, ry + rh / 2.0
                painter.drawLine(int(x0), int(cy), int(cx), int(cy))


class VisionInspectorWidget(QWidget):
    """Window-1 object list and overlay controls for the live visual scene."""

    def __init__(self, canvas: VisionCanvas, on_attention_select=None) -> None:
        super().__init__()
        self.canvas = canvas
        self._on_attention_select = on_attention_select
        self._objects: list[dict[str, Any]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.boxes = QCheckBox("Boxes"); self.boxes.setChecked(True)
        self.labels = QCheckBox("Labels"); self.labels.setChecked(True)
        self.confidence = QCheckBox("Confidence"); self.confidence.setChecked(True)
        self.track_ids = QCheckBox("Track IDs")
        self.track_ids.setChecked(True)
        self.motion = QCheckBox("Motion")
        for checkbox in (self.boxes, self.labels, self.confidence, self.track_ids, self.motion):
            checkbox.toggled.connect(self._apply_options)
            layout.addWidget(checkbox)

        self.legend = QLabel(
            f'<span style="color:{VISION_COLOR_FOCUSED}">● focus</span> · '
            f'<span style="color:{VISION_COLOR_NOTICED}">● noticed</span> · '
            f'<span style="color:{VISION_COLOR_MOTION}">● motion</span> · ? uncertain'
        )
        self.legend.setTextFormat(Qt.RichText)
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)

        self.freeze = QPushButton("Freeze frame")
        self.freeze.setCheckable(True)
        self.freeze.setToolTip("Hold the current frame and object map for visual labeling/teaching.")
        self.freeze.toggled.connect(self._freeze_toggled)
        layout.addWidget(self.freeze)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Object", "Conf", "Track"])
        self.tree.setRootIsDecorated(False)
        self.tree.itemClicked.connect(self._selected)
        layout.addWidget(self.tree, 1)
        self.detail = QLabel("No current objects")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self._apply_options()

    @property
    def frozen(self) -> bool:
        return self.freeze.isChecked()

    def _freeze_toggled(self, on: bool) -> None:
        self.freeze.setText("Frame frozen" if on else "Freeze frame")
        self.canvas.set_frozen(bool(on))
        self.detail.setText(
            "Frozen visual teaching frame. Select an object/region, then type a label such as: this is an eye."
            if on else
            (f"{len(self._objects)} current object{'s' if len(self._objects) != 1 else ''}" if self._objects else "No current objects")
        )

    def _apply_options(self, *_args) -> None:
        self.canvas.set_display_options(
            boxes=self.boxes.isChecked(),
            labels=self.labels.isChecked(),
            confidence=self.confidence.isChecked(),
            track_ids=self.track_ids.isChecked(),
            motion=self.motion.isChecked(),
        )

    def set_objects(self, objects: list[dict[str, Any]]) -> None:
        if self.frozen:
            return
        self._objects = self.canvas.decorate_objects([dict(item) for item in objects if isinstance(item, Mapping)])
        self.tree.clear()
        for obj in self._objects:
            label = str(obj.get("label") or "unknown")
            try:
                confidence = float(obj.get("confidence", 0.0) or 0.0)
                conf_text = f"{confidence:.2f}"
            except Exception:
                conf_text = "—"
            track_id = str(obj.get("track_id") or "")
            ref_text = str(obj.get("_ui_ref") or visual_ref_text(obj, confidence_threshold=VISION_CONFIDENT_THRESHOLD))
            item = QTreeWidgetItem([f"● {ref_text} {label}", conf_text, track_id[:16]])
            item.setData(0, Qt.UserRole, track_id)
            color = VisionCanvas._state_color(obj)
            item.setForeground(0, QBrush(color))
            item.setForeground(1, QBrush(color))
            self.tree.addTopLevelItem(item)
        self.detail.setText(f"{len(self._objects)} current object{'s' if len(self._objects) != 1 else ''}") if self._objects else self.detail.setText("No current objects")

    def _selected(self, item: QTreeWidgetItem, column: int) -> None:
        track_id = str(item.data(0, Qt.UserRole) or "")
        if not track_id:
            return
        self.canvas.flash_object(track_id)
        self._objects = self.canvas.decorate_objects(self._objects)
        obj = next((row for row in self._objects if str(row.get("track_id") or "") == track_id), None)
        for row in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(row)
            row_track = str(node.data(0, Qt.UserRole) or "")
            row_obj = next((entry for entry in self._objects if str(entry.get("track_id") or "") == row_track), None)
            if isinstance(row_obj, Mapping):
                color = VisionCanvas._state_color(row_obj)
                node.setForeground(0, QBrush(color))
                node.setForeground(1, QBrush(color))
        if callable(self._on_attention_select):
            snapshot = dict(obj) if isinstance(obj, Mapping) else {"track_id": track_id}
            snapshot["ui_frozen"] = bool(self.frozen)
            snapshot["ui_snapshot"] = self.canvas.snapshot_context()
            self._on_attention_select(track_id, snapshot)
        if not isinstance(obj, Mapping):
            return
        label = str(obj.get("label") or "unknown")
        status = str(obj.get("status") or "unknown")
        try:
            confidence = f"{float(obj.get('confidence', 0.0) or 0.0):.2f}"
        except Exception:
            confidence = "—"
        ref_text = str(obj.get("_ui_ref") or visual_ref_text(obj, confidence_threshold=VISION_CONFIDENT_THRESHOLD))
        state = str(obj.get("_ui_attention_state") or "noticed")
        self.detail.setText(f"{ref_text} · {label} · {state} · {status} · confidence {confidence}\n{track_id}")


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

        self.main_splitter = QSplitter(Qt.Vertical)
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.panels: dict[str, CompactablePanel] = {}

        self.vision = VisionCanvas()
        self.vision_inspector = VisionInspectorWidget(self.vision, self.controller.select_visual_object)
        self.body = BodyMapWidget()
        self.conversation = QPlainTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setPlaceholderText("Conversation / visible interaction")
        self.process_log = QPlainTextEdit()
        self.process_log.setReadOnly(True)
        self.process_log.setPlaceholderText("Process / component trace")
        self.process_log.setMaximumBlockCount(MAX_RAW_LINES)
        self.capability_strip = CapabilityStatusStrip("CAP")
        process_container = QWidget()
        process_layout = QVBoxLayout(process_container)
        process_layout.setContentsMargins(0, 0, 0, 0)
        process_layout.setSpacing(2)
        process_layout.addWidget(self.capability_strip)
        process_layout.addWidget(self.process_log, 1)

        self.panels["vision_inspector"] = CompactablePanel("Vision / objects", self.vision_inspector)
        self.panels["vision"] = CompactablePanel("Vision", self.vision)
        self.panels["body"] = CompactablePanel("Body / proprioception", self.body)
        self.panels["conversation"] = CompactablePanel("Conversation", self.conversation)
        self.panels["process"] = CompactablePanel("Process", process_container)

        self.top_splitter.addWidget(self.panels["vision_inspector"])
        self.top_splitter.addWidget(self.panels["vision"])
        self.top_splitter.addWidget(self.panels["body"])
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 4)
        self.top_splitter.setStretchFactor(2, 1)
        self.bottom_splitter.addWidget(self.panels["conversation"])
        self.bottom_splitter.addWidget(self.panels["process"])
        self.bottom_splitter.setStretchFactor(0, 3)
        self.bottom_splitter.setStretchFactor(1, 2)
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.bottom_splitter)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        layout.addWidget(self.main_splitter, 1)

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

    def save_workspace(self, settings: QSettings) -> None:
        settings.setValue("presence/main_splitter", _sizes_text(self.main_splitter))
        settings.setValue("presence/top_splitter", _sizes_text(self.top_splitter))
        settings.setValue("presence/bottom_splitter", _sizes_text(self.bottom_splitter))
        for key, panel in self.panels.items():
            settings.setValue(f"presence/panel/{key}/compact", panel.is_compact)

    def restore_workspace(self, settings: QSettings) -> None:
        for key, splitter in (
            ("presence/main_splitter", self.main_splitter),
            ("presence/top_splitter", self.top_splitter),
            ("presence/bottom_splitter", self.bottom_splitter),
        ):
            sizes = _settings_sizes(settings.value(key))
            if len(sizes) == splitter.count() and any(v > 0 for v in sizes):
                splitter.setSizes(sizes)
        for key, panel in self.panels.items():
            panel.set_compact(_settings_bool(settings.value(f"presence/panel/{key}/compact"), False))

    def _submit(self) -> None:
        text = self.input.text().strip()
        self.input.clear()
        if text:
            self.controller.submit_text(text)

    def append_conversation(self, line: str) -> None:
        self.conversation.appendPlainText(line)

    def append_process(self, line: str) -> None:
        self.process_log.appendPlainText(line)

    def update_snapshot(self, payload: Mapping[str, Any]) -> None:
        capability = payload.get("capability", {}) if isinstance(payload.get("capability"), Mapping) else {}
        self.capability_strip.update_payload(capability)
        current = self.panels["process"].status_label.text().split(" | cap ", 1)[0]
        self.panels["process"].set_status(f"{current} | {self.capability_strip.compact_text()}")

        vision = payload.get("vision", {}) if isinstance(payload.get("vision"), Mapping) else {}
        objects = vision.get("objects") if isinstance(vision.get("objects"), list) else []
        if objects and not self.vision_inspector.frozen:
            normalized = [dict(item) for item in objects if isinstance(item, Mapping)]
            self.vision.set_overlays(normalized)
            self.vision_inspector.set_objects(normalized)
            count = len(normalized)
            self.panels["vision_inspector"].set_status(f"{count} object{'s' if count != 1 else ''}")
            self.panels["vision"].set_status(f"{count} tracked")
        ref = str(vision.get("frame_ref") or "")
        if ref and not ref.startswith("ram:vision:"):
            self.vision.set_frame(ref)

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
        self.panels["body"].set_status(str(body.get("power_mode", "awake")))
        self.panels["process"].set_status(
            f"{pulse.get('thought_intent', 'idle')}/{pulse.get('thought_status', 'idle')} | "
            f"{self.capability_strip.compact_text()}"
        )

    def process(self, msg: UIMessage) -> None:
        self.body.update_event(msg)
        payload = msg.payload if isinstance(msg.payload, Mapping) else {}
        if msg.topic == "capability/state":
            self.capability_strip.update_payload(payload)
            current = self.panels["process"].status_label.text().split(" | cap ", 1)[0]
            self.panels["process"].set_status(f"{current} | {self.capability_strip.compact_text()}")
            return
        if msg.topic == "percept/vision":
            ref = str(payload.get("data_ref") or payload.get("frame_ref") or "")
            frame_bytes = payload.get("jpeg_bytes")
            width = int(payload.get("width") or 0)
            height = int(payload.get("height") or 0)
            if isinstance(frame_bytes, (bytes, bytearray)):
                self.vision.set_frame_bytes(frame_bytes, width, height, ref or "RAM frame")
                self.panels["vision"].set_status(f"RAM frame {payload.get('frame_id', '')}")
            elif ref and not ref.startswith("ram:vision:"):
                self.vision.set_frame(ref, width, height)
                self.panels["vision"].set_status(Path(ref).name)
            return

        if msg.topic == "ui/vision_current":
            objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
            normalized = [dict(item) for item in objects if isinstance(item, Mapping)]
            if not self.vision_inspector.frozen:
                self.vision.set_overlays(normalized)
                self.vision_inspector.set_objects(normalized)
            ref = str(payload.get("frame_ref") or "")
            if ref:
                self.vision.set_frame(ref)
            count = len(normalized)
            self.panels["vision_inspector"].set_status(f"{count} object{'s' if count != 1 else ''}")
            self.panels["vision"].set_status(f"{count} tracked")
            return


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
        self.docks: dict[str, CompactDockWidget] = {}

        self.raw = QPlainTextEdit(); self.raw.setReadOnly(True); self.raw.setMaximumBlockCount(MAX_RAW_LINES)
        self.detail = QPlainTextEdit(); self.detail.setReadOnly(True)
        self.evidence = QTreeWidget(); self.evidence.setHeaderLabels(["Type", "Reference", "From"])
        self.evidence.itemDoubleClicked.connect(self._open_evidence)
        self.organs = QPlainTextEdit(); self.organs.setReadOnly(True)
        self.runtime_log = LogTailWidget(Path(self.controller.bridge.memdir) / "logs" / "microbrain.log")
        self.tuning_tabs = self._build_tuning_tabs()
        self.slearn = SlearnJobWidget()
        self.heartbeat_strip = HeartbeatStatusStrip()
        self.capability_strip = CapabilityStatusStrip("CAP")
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.heartbeat_strip, 1)
        self.statusBar().addPermanentWidget(self.capability_strip, 2)

        self._dock("Raw event bus", self.raw, Qt.BottomDockWidgetArea, "raw_event_bus")
        self._dock("Runtime log", self.runtime_log, Qt.BottomDockWidgetArea, "runtime_log")
        self._dock("Selected event", self.detail, Qt.RightDockWidgetArea, "selected_event")
        self._dock("Evidence links", self.evidence, Qt.RightDockWidgetArea, "evidence_links")
        self._dock("DDNA / tuning / laws", self.tuning_tabs, Qt.LeftDockWidgetArea, "tuning")
        self._dock("Organ / bus status", self.organs, Qt.BottomDockWidgetArea, "organ_status")
        slearn_dock = self._dock("SLEARN / learning jobs", self.slearn, Qt.BottomDockWidgetArea, "slearn_jobs")
        self.slearn.set_status_callback(slearn_dock.set_compact_status)
        self._load_config_catalog()

    def _dock(
        self,
        title: str,
        widget: QWidget,
        area: Qt.DockWidgetArea,
        key: str,
    ) -> CompactDockWidget:
        dock = CompactDockWidget(title, widget, self)
        dock.setObjectName(f"dashboard_dock_{key}")
        self.addDockWidget(area, dock)
        self.docks[key] = dock
        return dock

    def save_workspace(self, settings: QSettings) -> None:
        settings.setValue("engineering/dock_state", self.saveState(WORKSPACE_SETTINGS_VERSION))
        for key, dock in self.docks.items():
            settings.setValue(f"engineering/dock/{key}/compact", dock.is_compact)

    def restore_workspace(self, settings: QSettings) -> None:
        state = settings.value("engineering/dock_state")
        if state is not None:
            try:
                self.restoreState(state, WORKSPACE_SETTINGS_VERSION)
            except Exception:
                pass
        for key, dock in self.docks.items():
            dock.set_compact(_settings_bool(settings.value(f"engineering/dock/{key}/compact"), False))

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
        self.organs.setPlainText(_pretty({k: payload.get(k) for k in ("queue_depth", "body_queue_depth", "dashboard_queue_depth", "dashboard_dropped_events", "neurons", "bus", "body_bus", "heartbeat", "adrenaline", "organs", "hypothesis_tuning", "release_tuning", "probe_runtime")}))
        heartbeat = payload.get("heartbeat", {}) if isinstance(payload.get("heartbeat"), Mapping) else {}
        adrenaline = payload.get("adrenaline", {}) if isinstance(payload.get("adrenaline"), Mapping) else {}
        self.heartbeat_strip.update_payloads(heartbeat, adrenaline)
        capability = payload.get("capability", {}) if isinstance(payload.get("capability"), Mapping) else {}
        self.capability_strip.update_payload(capability)
        slearn = payload.get("slearn", {}) if isinstance(payload.get("slearn"), Mapping) else {}
        self.slearn.update_snapshot(slearn)
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
        if _is_status_instrument_event(msg) or _is_ephemeral_visual_sample(msg):
            if msg.topic == "capability/state":
                payload = msg.payload if isinstance(msg.payload, Mapping) else {}
                self.capability_strip.update_payload(payload)
            # Current visual objects are a Window-1 instrument. Do not turn the
            # live object map into repeating Engineering trace/raw-event lines.
            return

        # SLEARN's workbench traffic belongs in its own engineering instrument,
        # not in the cognition trace. Meaningful learning/* result events still
        # continue through the normal trace after updating the job panel.
        if _is_slearn_diagnostic(msg):
            self.slearn.update_event(msg)
            return
        if msg.topic.startswith("learning/"):
            self.slearn.update_event(msg)

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
        self.settings = QSettings(WORKSPACE_SETTINGS_ORG, WORKSPACE_SETTINGS_APP)
        self.assistant_label, self.user_label = load_display_labels(memdir)
        self.transcript = TranscriptWriter(memdir, prefix="dashboard")
        self.presence = PresenceWindow(self)
        self.engineering = EngineeringWindow(self)
        self.timer = QTimer(); self.timer.timeout.connect(self._poll); self.timer.start(UI_POLL_MS)
        self.presence.append_conversation(f"{self.assistant_label} dashboard online. (/quit to close)")
        self.transcript.append_conversation(f"{self.assistant_label} dashboard online. (/quit to close)")
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.save_workspace)

    def save_workspace(self) -> None:
        self.settings.setValue("workspace/version", WORKSPACE_SETTINGS_VERSION)
        self.settings.setValue("presence/geometry", self.presence.saveGeometry())
        self.settings.setValue("engineering/geometry", self.engineering.saveGeometry())
        self.presence.save_workspace(self.settings)
        self.engineering.save_workspace(self.settings)
        self.settings.sync()

    def restore_workspace(self) -> bool:
        try:
            version = int(self.settings.value("workspace/version", 0) or 0)
        except Exception:
            version = 0
        if version != WORKSPACE_SETTINGS_VERSION:
            return False

        restored = False
        presence_geometry = self.settings.value("presence/geometry")
        engineering_geometry = self.settings.value("engineering/geometry")
        if presence_geometry is not None:
            try:
                restored = bool(self.presence.restoreGeometry(presence_geometry)) or restored
            except Exception:
                pass
        if engineering_geometry is not None:
            try:
                restored = bool(self.engineering.restoreGeometry(engineering_geometry)) or restored
            except Exception:
                pass
        self.presence.restore_workspace(self.settings)
        self.engineering.restore_workspace(self.settings)
        return restored

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

    def select_visual_object(self, track_id: str, object_snapshot: Mapping[str, Any] | None = None) -> None:
        """Treat an inspector click as pointing, not as labeling."""
        track_id = str(track_id or "").strip()
        if not track_id:
            return
        asyncio.create_task(self.bridge.select_visual_object(track_id, object_snapshot=object_snapshot))

    async def apply_runtime_tuning(self, key: str, value: str) -> None:
        try:
            record = await self.bridge.set_runtime_tuning(key, value)
            self.presence.append_conversation(f"status> tuning {record['key']}: {record['old']} → {record['new']}")
        except Exception as exc:
            self.presence.append_conversation(f"error> tuning change rejected: {exc}")

    def _conversation_line(self, msg: UIMessage, text: str, channel: str, source: str) -> str | None:
        if _is_slearn_diagnostic(msg):
            return None
        if msg.topic in {"ui/error", "control/error"}:
            return f"error> {text or _summary(msg.payload, 240)}"
        if msg.topic in {"ui/status", "control/status"}:
            return f"status> {text or _summary(msg.payload, 240)}"
        if msg.topic == "act/speech" and channel != "thought":
            return f"{self.assistant_label}> {text}"
        return None

    def _process_line(self, msg: UIMessage, text: str, channel: str, source: str) -> str | None:
        if _is_slearn_diagnostic(msg):
            return None
        if _is_status_instrument_event(msg) or _is_ephemeral_visual_sample(msg):
            return None
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
            origin = ""
            meta = msg.meta or {}
            payload = msg.payload if isinstance(msg.payload, Mapping) else {}
            origin = str(meta.get("origin") or payload.get("origin") or "").strip()
            return f"thought/probe[{origin}]> {text}" if origin else f"thought/probe> {text}"
        if msg.topic == "thought/probe":
            meta = msg.meta or {}
            payload = msg.payload if isinstance(msg.payload, Mapping) else {}
            origin = str(meta.get("origin") or payload.get("origin") or "").strip()
            return f"thought/probe[{origin}]> {text}" if origin else f"thought/probe> {text}"
        return f"{msg.topic}> {text}"

    def _poll(self) -> None:
        for msg in self.bridge.drain_nowait(limit=MAX_MESSAGES_PER_POLL):
            if not _is_ephemeral_visual_sample(msg):
                self.transcript.append_raw(msg)
            if msg.topic == "ui/pressure_state" and isinstance(msg.payload, Mapping):
                self.presence.update_pressure(msg.payload); continue
            if msg.topic == "ui/dashboard_snapshot" and isinstance(msg.payload, Mapping):
                self.presence.update_snapshot(msg.payload)
                self.engineering.update_snapshot(msg.payload)
                continue
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
    if not controller.restore_workspace():
        place_windows(controller.presence, controller.engineering)
    controller.presence.show(); controller.engineering.show()
    return controller
