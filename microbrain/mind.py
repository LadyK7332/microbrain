from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from microbrain.voice.tts import TTS
from microbrain.config import AppConfig
from microbrain.memory.emotional_journal import EmotionJournal
from microbrain.memory.memory_store import MemoryStore
from microbrain.hrm.core import HRMCore
from microbrain.pdna.core import PDNAStore
from microbrain.utils.logging_setup import configure_logging
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.orchestrator.neuron_loader import auto_register_neurons
from microbrain.orchestrator.debug_utils import set_debug_enabled
from microbrain.utils.whisper_audio import WhisperAudioListener, WhisperAudioConfig
from microbrain.utils.mic_probe_runtime import list_input_devices, probe_rms


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
    p.add_argument("--mic-device", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--vad-aggressiveness", type=int, default=2)
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

def _resolve_llama_backend(arg_backend: str, vulkan_flag: bool) -> str:
    """
    Normalize backend selection so we mirror CLI precedence:
    - --vulkan flag overrides everything
    - otherwise, use the explicit --llama-backend value (already seeded from env)
    """
    if vulkan_flag:
        return "vulkan"
    return arg_backend


def _resolve_memdir(arg_memdir: str | None) -> str:
    """
    CLI > MB_MEMDIR env > default ./memory
    """
    if arg_memdir:
        return arg_memdir
    env_memdir = os.getenv("MB_MEMDIR")
    if env_memdir:
        return env_memdir
    return str(Path.cwd() / "memory")


def _resolve_model_path(arg_model: str | None) -> str:
    """
    CLI > MB_LLAMA_MODEL env > auto-pick from ./models
    """
    explicit = arg_model or os.getenv("MB_LLAMA_MODEL")
    if explicit:
        return explicit
    return _pick_model(None)

async def main_async(cfg: AppConfig):
    logger = configure_logging(cfg.log_level)

    # Grab the running event loop so background threads (e.g. Vosk)
    # can safely schedule events into the orchestrator.
    loop = asyncio.get_running_loop()

    # Lazy imports to avoid circulars
    from microbrain.orchestrator.orchestrator import Orchestrator
    from microbrain.orchestrator.neuron_loader import auto_register_neurons
    from microbrain.llm_backend import llm_generate

    # Build orchestrator runtime
    orch = Orchestrator()
    vosk_listener = None  # will hold VoskAudioListener if voice is enabled

   # Optional: voice sink for speech output
    if cfg.voice:
        try:
            tts = TTS(
                rate=cfg.tts_rate,
                volume=cfg.tts_volume,
                preferred=cfg.tts_voice or "",
            )
            orch.kv_store["voice:tts"] = tts
            logger.info("Voice mode enabled (pyttsx3)")
        except Exception as exc:
            logger.error("Failed to initialize TTS; continuing without voice", exc_info=exc)
            orch.kv_store["voice:tts"] = None
            cfg.voice = False

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

    # --- Optional: Whisper mic listener -> percept/audio events ---
    whisper_listener = None  # will hold WhisperAudioListener if voice is enabled

    if cfg.voice:
        # --- Mic probe: enumerate + RMS/peak, fail loudly if too quiet ---
        try:
            devices = list_input_devices()
            logger.info("MIC PROBE: input devices (index | ch | default_sr | name):")
            for d in devices:
                logger.info(
                    "  %s | ch=%s | default_sr=%s | %s",
                    d["index"],
                    d["max_input_channels"],
                    int(d["default_samplerate"]),
                    d["name"],
                )

            logger.info(
                "MIC PROBE: selecting device=%s sample_rate=%s",
                cfg.mic_device,
                cfg.sample_rate,
            )

            res = probe_rms(
                device=cfg.mic_device,
                samplerate=cfg.sample_rate,
                seconds=0.75,
                rms_threshold=0.003,
            )
            logger.info(
                "MIC PROBE: OK | device=%s (%s) sr=%s rms=%.6f peak=%.6f",
                res.device,
                res.device_name,
                res.samplerate,
                res.rms,
                res.peak,
            )
        except Exception as exc:
            logger.error("MIC PROBE: FAILED | %s", exc)
            logger.error(
                "MIC PROBE: disabling voice. "
                "Try: pass --mic-device <index> (avoid Sound Mapper), "
                "disable exclusive mode on the mic, check privacy permissions."
            )
            cfg.voice = False

    # --- Whisper listener -> percept/audio ---
    if cfg.voice:
        def on_transcript(text: str) -> None:
            spoken = (text or "").strip()
            if not spoken:
                return

            # This callback runs on a background thread (Whisper listener thread),
            # so we must schedule the coroutine onto the main asyncio loop.
            logger.info("VOICE INJECT -> input/text | %r", spoken)

            asyncio.run_coroutine_threadsafe(
                orch.push_event(
                    "input/text",
                    spoken,
                    meta={"source": "mic", "channel": "repl"},
                ),
                loop,
            )
                
        try:
            wcfg = WhisperAudioConfig(
                model_name=cfg.whisper_model,
                device_index=cfg.mic_device,
                sample_rate=cfg.sample_rate,
                vad_aggressiveness=cfg.vad_aggressiveness,
            )
            whisper_listener = WhisperAudioListener(
                cfg=wcfg,
                on_transcript=on_transcript,
                on_debug=lambda s: logger.info("%s", s),
            )
            whisper_listener.start()
            logger.info(
                "Whisper audio listener started | model=%s sample_rate=%s device=%s vad=%s",
                cfg.whisper_model,
                cfg.sample_rate,
                cfg.mic_device,
                cfg.vad_aggressiveness,
            )
        except Exception as exc:
            logger.warning("Failed to start Whisper audio listener: %s", exc)
            whisper_listener = None

    # Start the orchestrator
    await orch.start()

    if cfg.voice:
        logger.info("Voice mode active … REPL disabled …")
        while True:
            await asyncio.sleep(0.1)

    else:
        logger.info("Starting text REPL …")
        while True:
            prompt = input("you> ").strip()
            if not prompt:
                continue

            await orch.push_event(
                "input/text",
                prompt,
                meta={"source": "cli", "channel": "repl"},
            )

            await orch.wait_for_idle(timeout=30.0)

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
        whisper_model=args.whisper_model,
        mic_device=args.mic_device,
        sample_rate=args.sample_rate,
        vad_aggressiveness=args.vad_aggressiveness,        
        tts_voice=args.tts_voice,
        tts_rate=args.tts_rate,
        tts_volume=args.tts_volume,
        log_level=args.log_level,
        voice=args.voice,
    )
    asyncio.run(main_async(cfg))

if __name__ == "__main__":
    main()
