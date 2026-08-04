from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from microbrain.utils.memdir import resolve_memdir_cli, ensure_child_dirs

logger = logging.getLogger(__name__)

from microbrain.config import AppConfig
from microbrain.memory.emotional_journal import EmotionJournal
from microbrain.memory.memory_store import MemoryStore
from microbrain.memory.mem_cell_store import MemCellStore
from microbrain.hrm.core import HRMCore
from microbrain.pdna.core import PDNAStore
from microbrain.pdna.access import profile_sections_for_kv
from microbrain.hormone import derive_ddna_modulators
from microbrain.utils.logging_setup import configure_logging
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.orchestrator.neuron_loader import auto_register_neurons
from microbrain.orchestrator.debug_utils import set_debug_enabled
from microbrain.ipc.token import ensure_token_file
from microbrain.utils.instance_lock import acquire_instance_lock
from microbrain.utils.heartbeat_stream import HEARTBEAT_HZ, HEARTBEAT_INTERVAL_S

_INSTANCE_LOCK = None
WhisperAudioListener = None
WhisperAudioConfig = None
from microbrain.utils.mic_probe_runtime import list_input_devices, probe_rms
from microbrain.llm_backend import llm_generate
from microbrain.babble_backend import babble_generate


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
    p.add_argument(
        "--debug-tail",
        action="store_true",
        help="(debug) With Textual, open a separate log-tail console. Dashboard already includes internal logs.",
    )
    p.add_argument("--ollama-base", default="http://localhost:11434")
    p.add_argument("--model", default="mistral")
    # --- LLM is opt-in (disabled by default) ---
    p.add_argument(
        "--llm",
        action="store_true",
        help="Enable the LLM reasoning pipeline (default: off).",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="Optional path to a .gguf model. Implies --llm and sets MB_LLAMA_MODEL.",
    )
    p.add_argument("--onnx-embed-path", default=None)
    p.add_argument("--onnx-provider", default=None)
    p.add_argument("--onnx-max-len", type=int, default=256)
    p.add_argument("--memdir", default=None)
    p.add_argument("--voice", action="store_true")
    p.add_argument("--mic-device", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--ui", choices=["repl", "textual", "dashboard"], default="textual")
    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--vad-aggressiveness", type=int, default=2)    
    
    p.add_argument("--tts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tts-voice", default="Zira")
    p.add_argument("--tts-rate", type=int, default=155)
    p.add_argument("--tts-volume", type=float, default=0.9)
    p.add_argument(
        "--mouth-backend",
        default="none",
        choices=["auto", "ipc", "local", "none"],
        help="Speech transport: auto prefers IPC briefly after spoken input, otherwise local TTS.",
    )
    p.add_argument(
        "--spoken-reply-bias-ttl",
        type=float,
        default=45.0,
        help="Seconds that recent spoken input biases replies toward IPC in auto mouth mode.",
    )
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
    base = resolve_memdir_cli(arg_memdir)

    # Ensure base + expected child folders exist
    child_dirs = [
        "_trash",
        "backup",
        "emotion",
        "motion",
        "sight",
        "sound",
        "state",
        "synapse",
        "thought",
        "touch",
        "episodes",
        "logs",
    ]

    ensure_child_dirs(base, child_dirs, logger=logger)

    logger.info("Resolved memdir: %s", str(base))
    return str(base)
def _resolve_model_path(arg_model: str | None) -> str:
    """
    CLI > MB_LLAMA_MODEL env > auto-pick from ./models
    """
    explicit = arg_model or os.getenv("MB_LLAMA_MODEL")
    if explicit:
        return explicit
    return _pick_model(None)


def publish_pdna_runtime_profile(orch: Orchestrator, pdna_profile: Any) -> None:
    """Publish active PDNA/DDNA profile sections into runtime KV.

    The active profile lives under the memdir (usually Z:/Memory/pdna_profile.json).
    This helper turns dormant genome sections into readable organs:
    pdna:affect_model, pdna:reinforcement_model, pdna:ddna_mutators,
    pdna:wans, and drive:ddna_modulators.
    """
    orch.kv_store["pdna:last"] = pdna_profile.to_dict() if hasattr(pdna_profile, "to_dict") else pdna_profile
    profile_sections = profile_sections_for_kv(pdna_profile)
    orch.kv_store["pdna:sections"] = profile_sections
    for section_name, section_value in profile_sections.items():
        orch.kv_store[f"pdna:{section_name}"] = section_value
    orch.kv_store["drive:ddna_modulators"] = derive_ddna_modulators(pdna_profile)


async def main_async(cfg: AppConfig):
    # Configure logging (configure_logging returns None; logger is module-level).
    ui_mode = getattr(cfg, 'ui', 'repl')
    log_file = None
    if getattr(cfg, 'memdir', None):
        log_file = str(Path(cfg.memdir) / 'logs' / 'microbrain.log')
    # In Textual mode, avoid printing logs to stdout (it can corrupt the TUI).
    configure_logging(cfg.log_level, log_file=log_file, console=(ui_mode not in {'textual', 'dashboard'}))

    # Grab the running event loop so background threads (e.g. Vosk)
    # can safely schedule events into the orchestrator.
    loop = asyncio.get_running_loop()

    # Lazy imports to avoid circulars
    from microbrain.orchestrator.orchestrator import Orchestrator
    from microbrain.orchestrator.neuron_loader import auto_register_neurons
    from microbrain.llm_backend import llm_generate

    # Build orchestrator runtime
    orch = Orchestrator()
    orch.kv_store["ui:mode"] = ui_mode
    vosk_listener = None  # will hold VoskAudioListener if voice is enabled

    # TTS output is independent from mic/STT. Configure the speech_output neuron via KV.
    orch.kv_store["tts:enabled"] = bool(getattr(cfg, "tts_enabled", False))
    orch.kv_store["tts:voice"] = getattr(cfg, "tts_voice", "Zira")
    orch.kv_store["tts:rate"] = int(getattr(cfg, "tts_rate", 155))
    orch.kv_store["tts:volume"] = float(getattr(cfg, "tts_volume", 0.9))
    orch.kv_store["mouth:enabled"] = False
    orch.kv_store["mouth:local_fallback"] = False
    
    # Legacy echo helper is default OFF so spoken input is not parroted back.
    orch.kv_store.setdefault("echo:enabled", False)
    mouth_backend = str(getattr(cfg, "mouth_backend", "none")).lower()
    orch.kv_store["mouth:enabled"] = mouth_backend != "none"
    orch.kv_store["speech:transport_mode"] = mouth_backend
    orch.kv_store["speech:default_transport"] = "none"
    orch.kv_store["speech:audio_preferred_transport"] = "none"
    orch.kv_store["speech:audio_bias_ttl_s"] = float(getattr(cfg, "spoken_reply_bias_ttl_s", 45.0))
    orch.kv_store.setdefault("interaction:last_input", {})
    orch.kv_store.setdefault("drive:need_signal:initiative", {})
    orch.kv_store.setdefault("rosehip:enabled", True)
    orch.kv_store.setdefault("rosehip:conversation_hold_s", 12.0)
    orch.kv_store.setdefault("rosehip:repeat_reply_window_s", 18.0)
    orch.kv_store.setdefault("rosehip:thought_min_interval_s", 35.0)
    orch.kv_store.setdefault("rosehip:clarify_min_interval_s", 30.0)
    if mouth_backend != "none":
        ensure_token_file(Path(cfg.memdir) / "ipc_token.txt")

    # Power schedule / sleep controls (OFF unless explicitly enabled).
    orch.kv_store.setdefault("power:sleep", False)
    orch.kv_store.setdefault("power:sleep_idle_s", 20.0)      # seconds since last external input
    orch.kv_store.setdefault("power:sleep_period_s", 30.0)    # seconds between consolidation cycles
    orch.kv_store.setdefault("power:sleep_kick", False)

    # Charging + schedule window (for future HW triggers)
    orch.kv_store.setdefault("power:charging", False)
    orch.kv_store.setdefault(
        "power:state",
        {"pct": 100.0, "charging": False, "sleep": False, "mode": "active", "last_ts": time.time()},
    )
    orch.kv_store.setdefault("power:battery_pct", 100.0)
    orch.kv_store.setdefault("power:mode", "active")
    orch.kv_store.setdefault("power:timezone", "America/Chicago")
    orch.kv_store.setdefault("power:charge_window_start", "22:00")
    orch.kv_store.setdefault("power:charge_window_end", "06:00")
    orch.kv_store.setdefault("power:schedule_enabled", True)
    orch.kv_store.setdefault("power:schedule_period_s", 10.0) # seconds between schedule checks
    orch.kv_store.setdefault("power:autosleep_on_charge", True)
    orch.kv_store.setdefault("power:last_external_ts", 0.0)
    orch.kv_store.setdefault("power:idle_enabled", True)
    orch.kv_store.setdefault("power:idle_after_s", 60.0)          # seconds idle before entering idle state
    orch.kv_store.setdefault("power:idle_cpu_threshold", 15.0)    # CPU% threshold (best effort; psutil if available)
    orch.kv_store.setdefault("power:busy_count", 0)
    orch.kv_store.setdefault("power:busy", False)
    orch.kv_store.setdefault("power:active_drain_per_sec", 100.0 / 86400.0)
    orch.kv_store.setdefault("power:idle_drain_per_sec", 40.0 / 86400.0)
    orch.kv_store.setdefault("power:sleep_drain_per_sec", 10.0 / 86400.0)
    orch.kv_store.setdefault("power:busy_drain_per_sec", 15.0 / 86400.0)
    orch.kv_store.setdefault("power:charge_per_sec", 100.0 / 3600.0)
    orch.kv_store.setdefault("power:reason_request_cost", 0.10)
    orch.kv_store.setdefault("power:reason_internal_cost", 0.03)
    orch.kv_store.setdefault("power:act_speech_cost", 0.12)
    orch.kv_store.setdefault("power:act_speech_thought_cost", 0.02)
    orch.kv_store.setdefault("power:task_start_cost", 0.15)
    orch.kv_store.setdefault("power:act_motor_cost", 0.50)
    # Power/need speech gate: body pressure updates status freely, but does not
    # seize the mouth unless critical or explicitly enabled by the operator.
    orch.kv_store.setdefault("drive:power:speech_gate_enabled", True)
    orch.kv_store.setdefault("drive:power:allow_unsolicited_speech", False)
    orch.kv_store.setdefault("drive:power:critical_speech_threshold", 0.90)
    orch.kv_store.setdefault("drive:power:status_cooldown_s", 120.0)
    orch.kv_store.setdefault("drive:power:last_status_ts", 0.0)
    orch.kv_store.setdefault("drive:power:last_status", {})
    orch.kv_store.setdefault("probe:enabled", True)
    orch.kv_store.setdefault("probe:every_s", 300.0)
    orch.kv_store.setdefault("probe:maintenance_every_s", 1800.0)
    orch.kv_store.setdefault("probe:activation_delta", 0.03)
    orch.kv_store.setdefault("probe:promotion_delta", 0.01)
    orch.kv_store.setdefault("mem_cell:now_hours", 36.0)
    orch.kv_store.setdefault("mem_cell:short_hours", 72.0)
    orch.kv_store.setdefault("mem_cell:long_hours", 96.0)
    orch.kv_store.setdefault("mem_cell:learned_hours", 336.0)

    # Vision proto-object defaults: unknowns should begin curious, not important.
    orch.kv_store.setdefault("vision:proto:enabled", True)
    orch.kv_store.setdefault("vision:proto:crop_px", 160)
    orch.kv_store.setdefault("vision:proto:min_repeat", 3)
    orch.kv_store.setdefault("vision:proto:max_tracks", 32)
    orch.kv_store.setdefault("vision:proto:stale_s", 45.0)
    orch.kv_store.setdefault("vision:proto:match_max_bits", 10)
    orch.kv_store.setdefault("vision:proto:focus_proximity", 0.18)
    orch.kv_store.setdefault("vision:proto:curiosity_start", 0.72)
    orch.kv_store.setdefault("vision:proto:importance_start", 0.15)
    orch.kv_store.setdefault("vision:proto:self_reinforce", 0.05)
    orch.kv_store.setdefault("vision:proto:question_threshold", 0.86)
    orch.kv_store.setdefault("vision:proto:question_cooldown_s", 90.0)
    orch.kv_store.setdefault("vision:proto:emit_internal_notes", True)
    orch.kv_store.setdefault("vision:proto:speak_questions", False)
    orch.kv_store.setdefault("concept:proto_confirm", 1)
    orch.kv_store.setdefault("vision:gaze_state", {"x": 0.5, "y": 0.5, "radius": 0.12, "mode": "roam"})
    orch.kv_store.setdefault("vision:gaze:budget_refill_per_s", 0.14)
    orch.kv_store.setdefault("vision:gaze:roam_dwell_s", 1.6)
    orch.kv_store.setdefault("vision:gaze:inspect_dwell_s", 1.0)
    orch.kv_store.setdefault("vision:gaze:settle_s", 0.22)
    orch.kv_store.setdefault("vision:gaze:emit_state_notes", False)
    orch.kv_store.setdefault("vision:workspace:ttl_s", 25.0)
    orch.kv_store.setdefault("vision:workspace:min_repeat", 3)
    orch.kv_store.setdefault("vision:workspace:commit_stability", 0.58)

    # Vision object-delta memory gate. Raw image/frame perception is parser input;
    # durable visual memory should normally be compact object/spatial deltas.
    orch.kv_store.setdefault("vision:raw_hrm_enabled", False)
    orch.kv_store.setdefault("vision:delta:enabled", True)
    orch.kv_store.setdefault("vision:delta:lidar_ttl_s", 1.5)
    orch.kv_store.setdefault("vision:delta:move_threshold_m", 0.15)
    orch.kv_store.setdefault("vision:delta:confidence_floor", 0.35)
    orch.kv_store.setdefault("vision:delta:base_required_voters", 2)
    orch.kv_store.setdefault("vision:delta:safe_space_required_voters", 3)
    orch.kv_store.setdefault("vision:delta:safe_places", ["home", "workshop", "base", "lab"])

    # Thought turn arbitration / drawer defaults. Thoughts are need expressions
    # bound to a scene. If their required components are available, they become
    # action candidates; if not, they wait in a demand-decaying drawer.
    orch.kv_store.setdefault("thought:turn:enabled", True)
    orch.kv_store.setdefault("thought:turn:base_recheck_s", 30.0)
    orch.kv_store.setdefault("thought:turn:user_recent_s", 90.0)
    orch.kv_store.setdefault("thought:turn:learning_rate", 0.05)
    orch.kv_store.setdefault(
        "thought:priority:base",
        {
            "power": 1.00,
            "safety": 0.96,
            "maintenance": 0.90,
            "human_uplift": 0.74,
            "task": 0.68,
            "self_need": 0.60,
            "social": 0.56,
            "novelty": 0.44,
            "expression": 0.34,
            "idle_thought": 0.18,
        },
    )
    orch.kv_store.setdefault("thought:priority:learned_modifiers", {})
    orch.kv_store.setdefault(
        "thought:turn:ttl_s",
        {
            "power": 900.0,
            "safety": 600.0,
            "maintenance": 1200.0,
            "human_uplift": 900.0,
            "task": 900.0,
            "self_need": 600.0,
            "social": 420.0,
            "novelty": 240.0,
            "expression": 180.0,
            "idle_thought": 120.0,
        },
    )

    # Capability circulation / metabolic glue defaults. This is a passive
    # lymph-like readiness layer: it reports possible routes, missing parts,
    # and redundant fallbacks, but it does not command action or speech.
    orch.kv_store.setdefault("capability:circulation:enabled", True)
    orch.kv_store.setdefault("capability:low_power_threshold_pct", 15.0)
    orch.kv_store.setdefault("capability:components", {})
    orch.kv_store.setdefault("capability:available_components", {})
    orch.kv_store.setdefault("capability:alias_available", {})
    orch.kv_store.setdefault("capability:state", {})
    orch.kv_store.setdefault("capability:fallbacks", {})

    # Body heartbeat / sympathetic scheduler.  The pacemaker itself is a fixed
    # infrastructure law; these KV values expose current runtime state and allow
    # cadence-profile tuning without turning the heartbeat into DDNA/personality.
    orch.kv_store.setdefault("body:heartbeat:hz", HEARTBEAT_HZ)
    orch.kv_store.setdefault("body:heartbeat:interval_s", HEARTBEAT_INTERVAL_S)
    orch.kv_store.setdefault("body:heartbeat:stats", {})
    orch.kv_store.setdefault("body:heartbeat:last", {})
    orch.kv_store.setdefault("body:heartbeat:missed_total", 0)
    orch.kv_store.setdefault("body:arousal_mode", "normal")
    orch.kv_store.setdefault("body:adrenaline", {})
    orch.kv_store.setdefault("body:organ_cadence", {})
    orch.kv_store.setdefault("body:cadence_profile:normal", {})
    orch.kv_store.setdefault("body:cadence_profile:alert", {})
    orch.kv_store.setdefault("body:cadence_profile:emergency", {})

    # Evidence recorder / hazard gating (split raw streams; OFF unless armed).
    orch.kv_store.setdefault("er:enabled", True)
    orch.kv_store.setdefault("er:armed", False)
    orch.kv_store.setdefault("er:manual_hold", False)
    orch.kv_store.setdefault("er:session_id", "")
    orch.kv_store.setdefault("er:session_dir", "")
    orch.kv_store.setdefault("er:preroll_s", 20.0)
    orch.kv_store.setdefault("er:postroll_s", 30.0)
    orch.kv_store.setdefault("er:hazard_threshold", 3)
    orch.kv_store.setdefault("er:capture_text", True)
    orch.kv_store.setdefault("er:last_trigger_ts", 0.0)
    orch.kv_store.setdefault("er:last_reason", "")
    orch.kv_store.setdefault("er:last_level", 0)
    orch.kv_store.setdefault("er:last_source", "")
    orch.kv_store.setdefault("er:audio_enabled", True)
    orch.kv_store.setdefault("er:visual_enabled", True)

    # --- Persistent memory wiring (MemoryStore + EmotionJournal) ---
    mem_store = MemoryStore(
        memdir=cfg.memdir,
        onnx_embed_path=cfg.onnx_embed_path,
        onnx_provider=cfg.onnx_provider,
        onnx_max_len=cfg.onnx_max_len,
    )
    orch.kv_store["memory:store"] = mem_store

    mem_root = cfg.memdir
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
    publish_pdna_runtime_profile(orch, pdna_store.profile)

    # Auto-load all neuron modules under microbrain.neurons.*
    # --- LLM enable flag (opt-in) ---
    llm_enabled = bool(getattr(cfg, "llm", False) or getattr(cfg, "llm_model", None))
    orch.kv_store["llm:enabled"] = llm_enabled
    if getattr(cfg, "llm_model", None):
        os.environ["MB_LLAMA_MODEL"] = cfg.llm_model

    auto_register_neurons(orch)

    if "speech_output" not in orch.neurons:
        logger.warning("speech_output neuron did not register; act/speech will not reach IPC or local TTS.")
    else:
        logger.info("speech_output neuron registered.")

    # Provide the LLM backend used by LLMReasonerNeuron
    #
    # If you pass --model none (or off), we run "babble backend" instead of an LLM.
    model_name = (cfg.model or "").strip().lower()

    backend_generate = babble_generate if model_name in ("none", "off", "babble") else llm_generate
    if backend_generate is babble_generate:
        logger.warning("LLM disabled; using babble backend for cognition output.")

    async def generate_with_state(prompt: str, meta: dict | None = None) -> str:
        m = dict(meta or {})

        # Pull boredom+attention gates from orchestrator state
        boredom = orch.kv_store.get("drive:boredom", {})
        if isinstance(boredom, dict):
            m["boredom_active"] = bool(boredom.get("active", False))
        else:
            m["boredom_active"] = bool(orch.kv_store.get("drive:boredom_active", False))

        m["allow_babble"] = bool(orch.kv_store.get("attention:allow_babble", True))

        out = backend_generate(prompt, m)
        import inspect
        if inspect.isawaitable(out):
            out = await out

        out = str(out or "")

        # Mirror into reason/output only for genuinely internal cognition.
        # User-facing repl/default/cli replies should stay on the speech path only.
        meta_channel = str(m.get("channel", "") or "").lower()
        if out and meta_channel in ("internal", "thought"):
            await orch.push_event(
                "reason/output",
                out,
                meta={"source": "cognition", "channel": "internal"},
            )

        logger.debug(
            "babble_probe",
            extra={
                "boredom_active": m.get("boredom_active"),
                "allow_babble": m.get("allow_babble"),
                "out_len": len(out),
                "prompt_len": len(prompt or ""),
            },
        )
        return out

    # --- LLM/backend wiring (opt-in) ---
    if orch.kv_store.get("llm:enabled", False):
        orch.kv_store["llm:generate"] = generate_with_state
    else:
        orch.kv_store.pop("llm:generate", None)
    
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
        try:
            from microbrain.utils.whisper_audio import WhisperAudioListener, WhisperAudioConfig
        except Exception:
            logger.exception("Voice mode requested but Whisper audio deps are missing.")
            WhisperAudioListener = None

        if WhisperAudioListener is None:
            logger.warning("Voice disabled: missing Whisper/VAD dependencies.")
        else:
            try:
                def _schedule(topic: str, payload: dict, meta: dict | None = None) -> None:
                    m = dict(meta or {})
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(
                            orch.push_event(
                                topic,
                                payload,
                                meta=m,
                                source="whisper",
                            )
                        )
                    )

                def _on_transcript(text: str) -> None:
                    audio_payload = {
                        "text": text,
                        "confidence": None,
                        "speaker": "user",
                        "channel": "repl",
                        "raw_meta": {
                            "input_modality": "audio",
                            "source": "mic",
                            "device_index": cfg.mic_device,
                        },
                    }
                    _schedule("percept/audio", audio_payload)

                def _on_utterance(text: str, pcm_bytes: bytes, sample_rate: int) -> None:
                    utt_payload = {
                        "text": text,
                        "pcm_bytes": pcm_bytes,
                        "sample_rate": int(sample_rate),
                        "channels": 1,
                        "speaker": "user",
                        "channel": "repl",
                        "raw_meta": {
                            "input_modality": "audio",
                            "source": "mic",
                            "device_index": cfg.mic_device,
                        },
                    }
                    _schedule("percept/audio_utterance", utt_payload)

                def _on_dbg(msg: str) -> None:
                    logger.debug("%s", msg)

                wcfg = WhisperAudioConfig(
                    model_name=str(getattr(cfg, "whisper_model", "small.en") or "small.en"),
                    device_index=getattr(cfg, "mic_device", None),
                    sample_rate=int(getattr(cfg, "sample_rate", 16000) or 16000),
                    vad_aggressiveness=int(getattr(cfg, "vad_aggressiveness", 2) or 2),
                    raw_only=False,
                )

                whisper_listener = WhisperAudioListener(
                    wcfg,
                    on_transcript=_on_transcript,
                    on_debug=_on_dbg,
                    on_audio_raw=None,
                    on_utterance=_on_utterance,
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

    # ------------------------------------------------------------------
    # Sidecars: background organs that must be explicitly started.
    # ------------------------------------------------------------------
    # Memory composer owns canonical mem-cell shard writes.  Other organs stage
    # pending rows, then this desk-goblin drains and composes them into memory.
    memory_composer_sidecar = None
    try:
        logger.info("Starting memory composer sidecar …")
        from microbrain.sidecars.memory_composer_sidecar import MemoryComposerSidecar

        memory_composer_sidecar = MemoryComposerSidecar(orch, memdir=cfg.memdir)
        await memory_composer_sidecar.start()
        orch.kv_store["mem_cell:composer:started"] = True
        logger.info("Memory composer sidecar started.")
    except Exception as exc:
        orch.kv_store["mem_cell:composer:started"] = False
        orch.kv_store["mem_cell:composer:error"] = repr(exc)
        logger.exception("Failed to start memory composer sidecar")

    # /read and /slearn are handled by the read sidecar so document chewing and
    # structured curriculum ingestion do not block the interaction/UI layer.
    read_sidecar = None
    try:
        logger.info("Starting read/SLEARN sidecar …")
        from microbrain.sidecars.read_sidecar import ReadSidecar

        read_sidecar = ReadSidecar(orch, memdir=cfg.memdir)
        await read_sidecar.start()
        orch.kv_store["read:sidecar_started"] = True
        orch.kv_store["slearn:sidecar_started"] = True
        logger.info("Read/slearn sidecar started.")
    except Exception as exc:
        orch.kv_store["read:sidecar_started"] = False
        orch.kv_store["slearn:sidecar_started"] = False
        orch.kv_store["slearn:sidecar_error"] = repr(exc)
        logger.exception("Failed to start read/slearn sidecar")

    # ------------------------------------------------------------------
    # Body pacemaker: one canonical 20-TPS infrastructure heartbeat.
    # ------------------------------------------------------------------
    # The pacemaker is its own body service. ``mind`` only starts it; raw body
    # cadence lives on the isolated infrastructure bus and never enters the
    # cognitive event stream.
    from microbrain.body.heartbeat import BodyHeartbeatPacemaker

    body_pacemaker = BodyHeartbeatPacemaker(orch)
    body_pacemaker.start()
    orch.kv_store["body:heartbeat:pacemaker"] = body_pacemaker

    if getattr(cfg, "ui", "repl") == "dashboard":
        logger.info("Starting native PySide6 dashboard …")
        from microbrain.ui.dashboard.qt_runtime import run_dashboard_frontend
        await run_dashboard_frontend(orch, memdir=cfg.memdir)
        return

    if getattr(cfg, "ui", "repl") == "textual":
        logger.info("Starting Textual UI …")
        from microbrain.ui.textual_bridge import run_textual_frontend
        await run_textual_frontend(orch, memdir=cfg.memdir)
        return

    if cfg.voice:
        logger.info("Voice mode active … REPL disabled …")
        while True:
            await asyncio.sleep(0.1)

    else:
        logger.info("Starting text REPL …")
        while True:
            # IMPORTANT: don't block the asyncio loop (lets body cadence + background outputs run)
            raw = await asyncio.to_thread(input, "you> ")
            prompt = (raw or "").strip()
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

    # If --debug is on, default log level should be DEBUG unless explicitly overridden.
    if getattr(args, "debug", False) and (getattr(args, "log_level", "WARNING").upper() == "WARNING"):
        args.log_level = "DEBUG"

    # Resolve memdir once so logging/memory/debug-tail all agree.
    args.memdir = _resolve_memdir(getattr(args, "memdir", None))

    # Hard guard: only one full MB body may own a memdir at a time.
    # Sidecars and organs share the current body; a second `python -m microbrain.mind`
    # against the same Z:\memory can corrupt/lock canonical memory shards.
    global _INSTANCE_LOCK
    try:
        _INSTANCE_LOCK = acquire_instance_lock(args.memdir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if getattr(args, "debug_tail", False):
        if not getattr(args, "debug", False):
            raise SystemExit("--debug-tail requires --debug")
        memdir = _resolve_memdir(getattr(args, "memdir", None))
        from microbrain.utils.debug_tail import tail_log, spawn_tail_window

        # Textual can use a separate tail window.  The native dashboard already
        # contains raw event and organ/log panels, so --debug-tail is redundant.
        if getattr(args, "ui", "repl") == "textual":
            spawned, reason = spawn_tail_window(args.memdir)
            if not spawned:
                logger.warning("debug-tail window spawn failed (%s); continuing without tail", reason)
        elif getattr(args, "ui", "repl") == "dashboard":
            logger.info("--debug-tail ignored: dashboard includes internal diagnostics")
        else:
            # REPL mode: act like a standalone tail command and exit.
            raise SystemExit(tail_log(args.memdir))

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
        tts_enabled=args.tts,
        tts_voice=args.tts_voice,
        tts_rate=args.tts_rate,
        tts_volume=args.tts_volume,
        log_level=args.log_level,
        voice=args.voice,
        llm=args.llm,
        llm_model=args.llm_model,
        ui=getattr(args, "ui", "repl"),
        mouth_backend=getattr(args, "mouth_backend", "auto"),
        spoken_reply_bias_ttl_s=float(getattr(args, "spoken_reply_bias_ttl", 45.0)),
    )

    if getattr(cfg, "ui", "repl") == "dashboard":
        from microbrain.ui.dashboard.qt_runtime import run_qt_async

        run_qt_async(main_async(cfg))
    else:
        asyncio.run(main_async(cfg))

if __name__ == "__main__":
    main()
