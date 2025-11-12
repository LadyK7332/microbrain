from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from microbrain.config import AppConfig
from microbrain.llamacpp_client import LlamaCppClient
from microbrain.llm.llama_runtime import ensure_llama_server
from microbrain.utils.logging_setup import configure_logging

try:
    from microbrain.regions.dense import DenseRegion  # CPU (optional)
except ModuleNotFoundError:
    DenseRegion = None  # fallback to GPU if CPU dense is missing


def build_arg_parser():
    p = argparse.ArgumentParser(description="Microbrain (split from monolith)")
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
    p.add_argument("--log-level", default="INFO")
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
    from microbrain.agent import Agent
    from microbrain.memory.memory_store import MemoryStore
    from microbrain.tools import ToolRegistry

    # Build runtime
    # --- Start / use llama.cpp server by default ---
    host = os.getenv("MB_LLAMA_HOST", getattr(cfg, "llama_host", "127.0.0.1"))
    port = int(os.getenv("MB_LLAMA_PORT", str(getattr(cfg, "llama_port", 8080))))
    model_path = _pick_model(os.getenv("MB_LLAMA_MODEL"))  # REQUIRED
    server_path = os.getenv("MB_LLAMA_SERVER", getattr(cfg, "llama_server", "")) or None
    threads = getattr(cfg, "threads", os.cpu_count() or 4)
    ngl = int(os.getenv("MB_LLAMA_NGL", "999"))
    extra = os.getenv("MB_LLAMA_ARGS", "")

    if not model_path:
        raise RuntimeError("Set MB_LLAMA_MODEL (or cfg.llama_model) to your .gguf model path.")

    # Start llama-server if not already up
    ensure_llama_server(
        model_path=model_path,
        server_path=server_path,
        host=host,
        port=port,
        threads=threads,
        ngl=ngl,
        extra_args=extra,
        wait_sec=int(os.getenv("MB_LLAMA_WAIT", "180")),  # was 30
    )

    # Build client (OpenAI-compatible first, legacy fallback handled inside)
    llm = LlamaCppClient(host=f"http://{host}:{port}", model=None)

    # Construct MemoryStore with our LLM client
    memdir = os.getenv("MB_MEMDIR", cfg.memdir)

    mem = MemoryStore(
        memdir=memdir,
        onnx_max_len=cfg.onnx_max_len,
        ollama=llm,  # name kept for compatibility
    )

    tools = ToolRegistry()
    agent = Agent(client=llm, memory=mem, tools=tools, logger=logger)

    # ---------------- voice/repl block MUST be inside main_async ----------------
    if cfg.voice:
        if not cfg.vosk_model_path:
            logger.error("Voice requested but --vosk-model-path is missing.")
            return
        from microbrain.voice.asr import STT
        from microbrain.voice.loop import voice_chat_loop
        from microbrain.voice.tts import TTS

        stt = STT(
            model_path=cfg.vosk_model_path,
            mic_device=cfg.mic_device,
            sample_rate=cfg.sample_rate,
            logger=logger,
        )
        tts = TTS(
            voice_name=cfg.tts_voice,
            rate=cfg.tts_rate,
            volume=cfg.tts_volume,
            logger=logger,
        )
        await voice_chat_loop(stt, tts, agent, logger)
    else:
        logger.info("Voice not enabled; starting text REPL. Ctrl+C to exit.")
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            reply = await agent.complete(prompt)
            print("bot>", reply)


def main():
    args = build_arg_parser().parse_args()
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
