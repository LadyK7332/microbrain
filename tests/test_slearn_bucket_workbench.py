from __future__ import annotations

import asyncio
from pathlib import Path

from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.neurons.syntax_learning_neuron import SyntaxLearningNeuron
from microbrain.orchestrator.neuron_base import NeuronConfig
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.sidecars.read_sidecar import ReadSidecar
from microbrain.sidecars.slearn_workbench import (
    SlearnWorkspaceCleaner,
    read_line_batch,
    scan_slearn_file,
)


def _processor() -> SyntaxLearningNeuron:
    return SyntaxLearningNeuron(
        NeuronConfig(name="test_slearn_processor", subscribed_topics=[], output_topics=[])
    )


def test_preflight_selects_normal_or_bucket_from_size_or_lines(tmp_path: Path):
    small = tmp_path / "small.slearn"
    small.write_text("IF TERM is one THEN CLASSIFY noun_concept\n" * 4, encoding="utf-8")
    normal = scan_slearn_file(small, bucket_min_bytes=10_000, bucket_min_lines=100)
    assert normal.mode == "normal"
    assert normal.line_count == 4

    by_lines = scan_slearn_file(small, bucket_min_bytes=10_000, bucket_min_lines=4)
    assert by_lines.mode == "bucket"
    assert by_lines.reason == "line_threshold"

    by_bytes = scan_slearn_file(small, bucket_min_bytes=1, bucket_min_lines=100)
    assert by_bytes.mode == "bucket"
    assert by_bytes.reason == "byte_threshold"


def test_streaming_line_batch_resumes_from_byte_cursor(tmp_path: Path):
    path = tmp_path / "many.slearn"
    path.write_text("".join(f"line {i}\n" for i in range(1, 11)), encoding="utf-8")

    first = read_line_batch(path, byte_offset=0, line_number=0, max_lines=4)
    assert [text for _, text in first.lines] == ["line 1", "line 2", "line 3", "line 4"]
    assert not first.eof

    second = read_line_batch(path, byte_offset=first.byte_offset, line_number=first.end_line, max_lines=4)
    assert [text for _, text in second.lines] == ["line 5", "line 6", "line 7", "line 8"]
    assert second.start_line == 5


def test_bulk_rule_batch_stages_one_file_and_uses_stable_rule_ids(tmp_path: Path):
    memdir = tmp_path / "mem"
    store = MemCellStore(memdir, composer_enabled=True, writer_id="bucket-test")
    processor = _processor()
    items = [
        {
            "teaching_note": f'IF TERM is "word{i}" THEN CLASSIFY noun_concept, category_test',
            "source_name": "bulk.slearn",
            "source_path": str(tmp_path / "bulk.slearn"),
            "source_line": i + 1,
        }
        for i in range(32)
    ]

    result = processor.apply_slearn_batch(store, items, weight=3)
    assert result["accepted"] == 32
    assert result["rejected"] == 0
    assert len(result["staged_paths"]) == 1
    # Pure classifier bulk staging should not need to load the canonical tier.
    assert "learned" not in store._tier_loaded

    # Restart-stable identity: the exact rule gets the same cell id even though
    # source line/time metadata differ.
    parsed_a = processor._parse_teaching_note(items[0]["teaching_note"])
    parsed_b = processor._parse_teaching_note(items[0]["teaching_note"])
    parsed_a.update({"reinforce_weight": 3, "source_mode": "slearn", "source_line": 1, "ts": 1.0})
    parsed_b.update({"reinforce_weight": 3, "source_mode": "slearn", "source_line": 999, "ts": 999.0})
    assert processor._build_rule_cell(parsed_a)["id"] == processor._build_rule_cell(parsed_b)["id"]


def test_workspace_cleaner_restores_active_floor_and_quarantines_unknown_job(tmp_path: Path):
    cleaner = SlearnWorkspaceCleaner(tmp_path / "mem")
    unknown = cleaner.root / "job-old"
    unknown.mkdir(parents=True, exist_ok=True)
    (unknown / "job.json").write_text("{}", encoding="utf-8")

    baseline = cleaner.snapshot()
    prepared = cleaner.prepare(job_id="job-current", source_path=tmp_path / "lesson.slearn")
    assert prepared["clean"]
    assert not unknown.exists()
    assert any(cleaner.quarantine.iterdir())

    finished = cleaner.finish(job_id="job-current", baseline=baseline)
    assert finished["clean"]
    assert finished["baseline_restored"]


def test_read_sidecar_bucket_path_does_not_emit_one_control_event_per_rule(tmp_path: Path):
    async def run() -> list[str]:
        memdir = tmp_path / "mem"
        slearn_dir = memdir / "slearn_dir"
        slearn_dir.mkdir(parents=True)
        sheet = slearn_dir / "bulk.slearn"
        sheet.write_text(
            "# MB_SLEARN\n"
            + "".join(
                f'IF TERM is "word{i}" THEN CLASSIFY noun_concept, category_test\n'
                for i in range(24)
            ),
            encoding="utf-8",
        )

        orch = Orchestrator()
        orch.kv_store["slearn:enabled"] = True
        orch.kv_store["slearn:dir"] = str(slearn_dir)
        orch.kv_store["slearn:idle_after_s"] = 0.0
        orch.kv_store["slearn:bucket_min_lines"] = 10
        orch.kv_store["slearn:bucket_min_bytes"] = 10_000_000
        orch.kv_store["slearn:bucket_batch_lines"] = 12
        orch.kv_store["mem_cell:composer:started"] = True
        sidecar = ReadSidecar(orch, memdir=str(memdir))

        # pass 1 = preflight, pass 2 = first bucket
        await sidecar._run_slearn_once()
        await sidecar._run_slearn_once()

        topics: list[str] = []
        while not orch.event_queue.empty():
            topics.append(orch.event_queue.get_nowait().topic)
        return topics

    topics = asyncio.run(run())
    assert "slearn/preflight" in topics
    assert "slearn/progress" in topics
    assert "control/slearn" not in topics


def test_bucket_job_waits_for_durable_composer_commit_before_completion(tmp_path: Path):
    from microbrain.memory.mem_cell_composer import MemCellComposer

    async def run() -> tuple[bool, list[str], dict]:
        memdir = tmp_path / "mem"
        slearn_dir = memdir / "slearn_dir"
        slearn_dir.mkdir(parents=True)
        sheet = slearn_dir / "lesson.slearn"
        sheet.write_text(
            "".join(
                f'IF TERM is "word{i}" THEN CLASSIFY noun_concept, category_test\n'
                for i in range(12)
            ),
            encoding="utf-8",
        )

        orch = Orchestrator()
        orch.kv_store.update({
            "slearn:enabled": True,
            "slearn:dir": str(slearn_dir),
            "slearn:idle_after_s": 0.0,
            "slearn:bucket_min_lines": 10,
            "slearn:bucket_min_bytes": 10_000_000,
            "slearn:bucket_batch_lines": 100,
            "mem_cell:composer:started": True,
        })
        sidecar = ReadSidecar(orch, memdir=str(memdir))

        await sidecar._run_slearn_once()  # preflight
        await sidecar._run_slearn_once()  # stage + EOF, must wait for receipt
        assert sheet.exists()
        assert orch.kv_store.get("slearn:status") == "waiting_commit"

        MemCellComposer(memdir).compose_once()
        await sidecar._run_slearn_once()  # commit observed -> cleanup + move + completion

        topics: list[str] = []
        while not orch.event_queue.empty():
            topics.append(orch.event_queue.get_nowait().topic)
        state = sidecar._load_slearn_state_file(slearn_dir)
        return (slearn_dir / "ready" / "lesson.slearn").exists(), topics, state

    moved, topics, state = asyncio.run(run())
    assert moved
    assert "slearn/completed" in topics
    assert "learning/completed" in topics
    assert state.get("active_file") == ""
    assert state.get("last_result", {}).get("baseline_restored") is True


def test_slearn_crash_recovery_detaches_foreign_and_prior_attempt_receipts(tmp_path: Path):
    async def run() -> tuple[dict, str, str, list[str]]:
        memdir = tmp_path / "mem"
        slearn_dir = memdir / "slearn_dir"
        slearn_dir.mkdir(parents=True)
        sheet = slearn_dir / "current.slearn"
        sheet.write_text(
            "".join(
                f'IF TERM is "current{i}" THEN CLASSIFY noun_concept, category_current\n'
                for i in range(4)
            ),
            encoding="utf-8",
        )

        orch = Orchestrator()
        orch.kv_store.update({
            "slearn:enabled": True,
            "slearn:dir": str(slearn_dir),
            "slearn:idle_after_s": 0.0,
            "slearn:bucket_min_lines": 1,
            "slearn:bucket_min_bytes": 10_000_000,
            "slearn:bucket_batch_lines": 1,
            "mem_cell:composer:started": True,
        })
        sidecar = ReadSidecar(orch, memdir=str(memdir))

        await sidecar._run_slearn_once()  # establish the current job/attempt
        state = sidecar._load_slearn_state_file(slearn_dir)
        job_id = str(state["job_id"])
        started_at = float(state["started_at"])

        # Receipt from another SLEARN job: explicit writer ownership mismatch.
        foreign_store = MemCellStore(
            memdir,
            composer_enabled=True,
            writer_id="slearn-sidecar-job-foreign",
        )
        foreign_store.stage_cells([{"id": "foreign-cell", "kind": "test"}], tier="learned")
        foreign_receipt = foreign_store.take_staged_paths("learned")[0]

        # Receipt from an older attempt of the *same stable job*. stable_job_id is
        # intentionally restart-stable, so queued_at must distinguish attempts.
        old_attempt_store = MemCellStore(
            memdir,
            composer_enabled=True,
            writer_id=f"slearn-sidecar-{job_id}",
        )
        old_attempt_store.stage_cells([{"id": "old-attempt-cell", "kind": "test"}], tier="learned")
        old_attempt_receipt = old_attempt_store.take_staged_paths("learned")[0]
        old_path = Path(old_attempt_receipt)
        lines = old_path.read_text(encoding="utf-8").splitlines()
        import json
        envelope = json.loads(lines[0])
        envelope["queued_at"] = started_at - 3600.0
        lines[0] = json.dumps(envelope, ensure_ascii=False)
        old_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        state["receipts"] = [foreign_receipt, old_attempt_receipt]
        sidecar._save_slearn_state_file(slearn_dir, state)

        await sidecar._run_slearn_once()
        recovered = sidecar._load_slearn_state_file(slearn_dir)

        topics: list[str] = []
        while not orch.event_queue.empty():
            topics.append(orch.event_queue.get_nowait().topic)
        return recovered, foreign_receipt, old_attempt_receipt, topics

    state, foreign_receipt, old_attempt_receipt, topics = asyncio.run(run())
    receipts = list(state.get("receipts", []) or [])
    assert foreign_receipt not in receipts
    assert old_attempt_receipt not in receipts
    # Recovery detaches ownership only; it does not delete possibly valid pending
    # memory work from an interrupted job. The composer may still commit it.
    assert Path(foreign_receipt).exists()
    assert Path(old_attempt_receipt).exists()
    recovery = dict(state.get("receipt_recovery", {}) or {})
    assert recovery.get("last_detached") == 2
    assert recovery.get("last_reasons") == {"foreign_job": 1, "older_attempt": 1}
    assert "slearn/recovery" in topics
