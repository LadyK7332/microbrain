from pathlib import Path

from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.sidecars.read_sidecar import ReadSidecar


def test_slearn_extension_chunks_as_text(tmp_path):
    sheet = tmp_path / "lesson.slearn"
    sheet.write_text(
        "# MB_SLEARN\n"
        "IF USER asks minecraft THEN CLASSIFY question_minecraft AND REPLY A voxel adventure game.\n",
        encoding="utf-8",
    )

    sidecar = ReadSidecar(Orchestrator(), memdir=str(tmp_path / "mem"))
    chunk = sidecar._chunk_for(sheet, 0, chunk_lines=80, chunk_chars=2400)

    assert chunk is not None
    assert chunk.kind == "text"
    rules = sidecar._extract_slearn_rules(chunk.text)
    assert len(rules) == 1
    assert rules[0][1].startswith("IF USER asks minecraft THEN")


def test_slearn_candidate_list_keeps_slearn_files(tmp_path):
    (tmp_path / "lesson.slearn").write_text("IF NEED exists THEN CLASSIFY need_object\n", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ready").mkdir()

    sidecar = ReadSidecar(Orchestrator(), memdir=str(tmp_path / "mem"))
    candidates = sidecar._list_slearn_candidates(tmp_path)

    assert [p.name for p in candidates] == ["lesson.slearn"]
