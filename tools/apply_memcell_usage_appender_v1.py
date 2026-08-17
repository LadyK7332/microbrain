from __future__ import annotations

"""Apply MemCell temporary usage appender v1 to the current working tree.

This patcher edits ``microbrain/memory/mem_cell_store.py`` by anchors so it can
be run after nearby repository updates without replacing the whole file.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "microbrain" / "memory" / "mem_cell_store.py"

IMPORT_ANCHOR = "from microbrain.language_scaffold import analyze_english_structure\n"
IMPORT_PATCH = IMPORT_ANCHOR + "from microbrain.memory.mem_cell_usage_appender import attach_temporary_usage_appendix, merge_meta_with_usage_appendix\n"

MERGE_OLD = '''        meta = dict(old.get("meta", {}) or {})
        meta.update(dict(new.get("meta", {}) or {}))
        merged["meta"] = meta
        return merged
'''
MERGE_NEW = '''        meta = merge_meta_with_usage_appendix(old.get("meta", {}), new.get("meta", {}), now_ts=now_ts)
        merged["meta"] = meta
        return merged
'''

APPEND_ANCHOR = '''        # back-link strongest immediate pieces into utterance. Raw tokens remain
        # evidence; word roles and thought templates are the reusable scaffold.
'''
APPEND_BLOCK = '''        # Temporary usage appendices: record how each mem-cell was used in this
        # sentence without turning that single use into durable truth. These
        # atoms let word-role/pattern cells self-update by repeated usage later.
        if not bool((meta or {}).get("skip_usage_appendix", False)):
            usage_appended_cells = attach_temporary_usage_appendix(
                utterance=utterance,
                token_cells=token_cells,
                word_role_cells=word_role_cells,
                thought_template_cells=thought_template_cells,
                clause_frame_cells=clause_frame_cells,
                learning_frame_cells=learning_frame_cells,
                general_pattern_cells=general_pattern_cells,
                linker_cells=linker_cells,
                text=text,
                role=role,
                topic=topic,
                source=source,
                structure=language_structure,
                meta=meta,
            )
            for usage_cell in usage_appended_cells:
                self.upsert_cell(usage_cell, tier=tier, touch=False, flush=False)

''' + APPEND_ANCHOR


def patch_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    if old not in text:
        raise SystemExit(f"Could not find anchor for {label}. File may have changed; inspect {TARGET}.")
    return text.replace(old, new, 1), True


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing target: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    changed = []

    text, did = patch_once(text, IMPORT_ANCHOR, IMPORT_PATCH, "usage appender import")
    if did:
        changed.append("import")

    text, did = patch_once(text, MERGE_OLD, MERGE_NEW, "metadata usage-appendix merge")
    if did:
        changed.append("merge_meta")

    text, did = patch_once(text, APPEND_ANCHOR, APPEND_BLOCK, "ingest_text usage appendix hook")
    if did:
        changed.append("ingest_hook")

    TARGET.write_text(text, encoding="utf-8")
    if changed:
        print("Applied memcell usage appender v1:", ", ".join(changed))
    else:
        print("Memcell usage appender v1 already applied.")


if __name__ == "__main__":
    main()
