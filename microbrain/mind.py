from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from microbrain.config import AppConfig
from microbrain.llamacpp_client import LlamaCppClient
from microbrain.llm.llama_runtime import ensure_llama_server
from microbrain.memory.emotional_journal import EmotionJournal
from microbrain.memory.memory_store import MemoryStore
from microbrain.hrm.core import HRMCore
from microbrain.pdna.core import PDNAStore
from microbrain.utils.logging_setup import configure_logging
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.orchestrator.neuron_loader import auto_register_neurons
from microbrain.orchestrator.debug_utils import set_debug_enabled

try:
    from microbrain.regions.dense import DenseRegion  # CPU (optional)
except ModuleNotFoundError:
    DenseRegion = None  # fallback to GPU if CPU dense is missing


def build_arg_parser():
    p = argparse.ArgumentParser(description="Microbrain (split from monolith)")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging and neuron debug output.",
    )
    p.add_argument("--ollama-base", default="http://localhost:11434")
    p.add_argument("--model", default="mistral")
    p.add_argument("--onnx-embed-path", default=None)
    p.add_argument("--onnx-provider", default=None)
    p.add_argument("--onnx-max-len", type=int, default=256)
    p.add_argument("--memdir", default=None)
    p.add_argument("--voice", action="store_true")
    p.add_argument("--vosk-model-path", default=None)
    p.add_argument("--mic-device", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--tts-voice", default=None)
    p.add_argument("--tts-rate", type=int, default=170)
    p.add_argument("--tts-volume", type=float, default=1.0)
    p.add_argument(
        "--log-level",
        default="WARNING",
        help="Log level: DEBUG, INFO, WARNING, ERROR (default: WARNING).",
    )
    p.add_argument(
        "--llama-backend",
        default=os.getenv("MB_LLAMA_BACKEND", "auto"),
        choices=["auto", "vulkan", "cuda", "metal", "rocm", "cpu"],
        help="Backend for llama.cpp (auto|vulkan|cuda|metal|rocm|cpu).",
    )
    p.add_argument("--vulkan", action="store_true", help="Alias for --llama-backend vulkan")

    return p


# scan .gguf model and present menu
def _scan_gguf(dirpath: Path) -> list[Path]:
    try:
        return sorted(
            dirpath.glob("*.gguf"),
            key=lambda p: p.stat().st_size,
            reverse=True,  # biggest first
        )
    except Exception:
        return []


# if session is interactive then show menu
def _pick_model(model_env: str | None) -> str:
    # 1) Respect explicit env if it points to a real file
    if model_env and Path(model_env).exists():
        return model_env

    # 2) Scan default models directory next to this file
    models_dir = Path(__file__).resolve().parent / "models"
    candidates = _scan_gguf(models_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No .gguf models found in {models_dir}. "
            "Set MB_LLAMA_MODEL or place a .gguf in microbrain/models."
        )

    # 3) Optional: MB_LLAMA_AUTOPICK=N (1-based index)
    auto = os.getenv("MB_LLAMA_AUTOPICK")
    if auto:
        try:
            idx = int(auto)
            if 1 <= idx <= len(candidates):
                return str(candidates[idx - 1])
        except Exception:
            pass  # fall through to interactive/default

    # 4) If interactive TTY and multiple, present a tiny menu
    if sys.stdin.isatty() and len(candidates) > 1:
        print("\nSelect a GGUF model:")
        for i, p in enumerate(candidates, 1):
            try:
                sz_gib = p.stat().st_size / (1024**3)
                print(f"[{i}] {p.name}  ({sz_gib:.2f} GiB)")
            except Exception:
                print(f"[{i}] {p.name}")
        choice = input("Enter number [1]: ").strip()
        if choice.isdigit():
            ch = int(choice)
            if 1 <= ch <= len(candidates):
                return str(candidates[ch - 1])

    # 5) Default to the first (largest) candidate
    return str(candidates[0])

    #####


async def main_async(cfg: AppConfig):
    logger = configure_logging(cfg.log_level)

    # Lazy imports to avoid circulars
    from microbrain.orchestrator.orchestrator import Orchestrator
    from microbrain.orchestrator.neuron_loader import auto_register_neurons
    from microbrain.llm_backend import llm_generate

    # Build orchestrator runtime
    orch = Orchestrator()

    # --- Persistent memory wiring (MemoryStore + EmotionJournal) ---
    mem_store = MemoryStore(
        memdir=cfg.memdir,
        onnx_embed_path=cfg.onnx_embed_path,
        onnx_provider=cfg.onnx_provider,
        onnx_max_len=cfg.onnx_max_len,
    )
    orch.kv_store["memory:store"] = mem_store

    mem_root = cfg.memdir or os.getenv("MB_MEMDIR") or str(Path.cwd() / "memory")
    ej_path = Path(mem_root) / "emotion_journal.jsonl"
    ejournal = EmotionJournal(str(ej_path))
    orch.kv_store["memory:emotion_journal"] = ejournal

    # --- HRM core wiring (concept graph + synapses) ---
    hrm_core = HRMCore(memdir=mem_root)
    orch.kv_store["hrm:core"] = hrm_core

    # --- PDNA wiring (personality DNA) ---
    pdna_store = PDNAStore(memdir=mem_root, profile_name="microbrain_default")
    orch.kv_store["pdna:store"] = pdna_store
    orch.kv_store["pdna:profile"] = pdna_store.profile

    # Auto-load all neuron modules under microbrain.neurons.*
    auto_register_neurons(orch)

    # Provide the LLM backend used by LLMReasonerNeuron
    orch.kv_store["llm:generate"] = llm_generate


    # Start the orchestrator
    await orch.start()

    # For now, voice isn't wired into the orchestrator, so we always run text REPL.
    if cfg.voice:
        logger.warning(
            "Voice mode is not yet integrated with the orchestrator; "
            "falling back to text REPL."
        )

    logger.info("Starting text REPL via orchestrator. Ctrl+C to exit.")

    try:
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not prompt:
                continue

            # Send user text into the orchestrator as input/text
            await orch.push_event(
                "input/text",
                prompt,
                meta={"source": "cli", "channel": "repl"},
            )

            # Let neurons process; speech_output neuron will print replies
            await orch.wait_for_idle(timeout=30.0)
    finally:
        await orch.stop()
        logger.info("MicroBrain orchestrator stopped.")

def main():
    args = build_arg_parser().parse_args()

    # NEW: flip the global neuron debug flag based on CLI
    set_debug_enabled(getattr(args, "debug", False))

    cfg = AppConfig(
        onnx_embed_path=args.onnx_embed_path,
        onnx_provider=args.onnx_provider,
        onnx_max_len=args.onnx_max_len,
        ollama_base=args.ollama_base,
        model=args.model,
        memdir=args.memdir,
        vosk_model_path=args.vosk_model_path,
        mic_device=args.mic_device,
        sample_rate=args.sample_rate,
        tts_voice=args.tts_voice,
        tts_rate=args.tts_rate,
        tts_volume=args.tts_volume,
        log_level=args.log_level,
        voice=args.voice,
    )
    asyncio.run(main_async(cfg))

if __name__ == "__main__":
    main()
