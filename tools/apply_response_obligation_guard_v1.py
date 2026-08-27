from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()
TARGET = ROOT / "microbrain" / "neurons" / "native_responder_neuron.py"
IMPORT_LINE = "from microbrain.response_obligation_guard import guard_native_response\n"
ANCHOR_IMPORT = "from microbrain.orchestrator.orchestrator import Orchestrator\n"

OLD_REPLY_BLOCK = '''        reply = await self._build_response(
            ctx,
            text=text,
            shape=shape,
            payload=payload,
            mem_store=mem_store,
            syntax_guidance=syntax_guidance,
        )
        if not reply:
            return []
'''

NEW_REPLY_BLOCK = '''        reply = await self._build_response(
            ctx,
            text=text,
            shape=shape,
            payload=payload,
            mem_store=mem_store,
            syntax_guidance=syntax_guidance,
        )

        guard_result = guard_native_response(
            user_text=text,
            proposed_reply=reply,
            shape=shape,
            payload=payload,
            raw_meta=raw_meta,
            syntax_guidance=syntax_guidance,
        )
        await ctx.set_kv("response_obligation:last", guard_result.to_dict())
        if guard_result.action != "accept":
            await ctx.log_debug(
                f"[{self.name}] Response obligation guard",
                action=guard_result.action,
                reason=guard_result.reason,
                original=guard_result.original_text[:120],
                proposed=guard_result.proposed_text[:120],
                final=guard_result.text[:120],
            )
        reply = guard_result.text
        if not reply:
            return []
'''

OLD_LAST_BLOCK = '''                "transport_source": transport_source,
            },
        )
'''

NEW_LAST_BLOCK = '''                "transport_source": transport_source,
                "response_obligation": guard_result.to_dict(),
            },
        )
'''

OLD_PAYLOAD_BLOCK = '''                    "style": "assistant",
                    "memory_cell_ids": memory_cell_ids,
                },
'''

NEW_PAYLOAD_BLOCK = '''                    "style": "assistant",
                    "memory_cell_ids": memory_cell_ids,
                    "response_obligation": guard_result.to_dict(),
                },
'''

OLD_META_BLOCK = '''                    "shape": shape,
                    "memory_cell_ids": memory_cell_ids,
                },
'''

NEW_META_BLOCK = '''                    "shape": shape,
                    "memory_cell_ids": memory_cell_ids,
                    "response_obligation": guard_result.to_dict(),
                },
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Could not apply {label}: expected 1 match, found {count}.")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing target file: {TARGET}")
    text = TARGET.read_text(encoding="utf-8")

    if IMPORT_LINE not in text:
        if ANCHOR_IMPORT not in text:
            raise SystemExit("Could not find import anchor in native_responder_neuron.py")
        text = text.replace(ANCHOR_IMPORT, ANCHOR_IMPORT + IMPORT_LINE, 1)

    text = replace_once(text, OLD_REPLY_BLOCK, NEW_REPLY_BLOCK, "response guard call")
    text = replace_once(text, OLD_LAST_BLOCK, NEW_LAST_BLOCK, "native_responder:last trace")
    text = replace_once(text, OLD_PAYLOAD_BLOCK, NEW_PAYLOAD_BLOCK, "speech payload trace")
    text = replace_once(text, OLD_META_BLOCK, NEW_META_BLOCK, "speech meta trace")

    TARGET.write_text(text, encoding="utf-8")
    print("Applied Response Obligation Guard v1 to microbrain/neurons/native_responder_neuron.py")


if __name__ == "__main__":
    main()
