from __future__ import annotations

from dataclasses import asdict, dataclass

from microbrain.utils.memdir import resolve_memdir_cli


@dataclass
class AppConfig:
    onnx_embed_path: str | None = None
    onnx_provider: str | None = None
    onnx_max_len: int = 256
    ollama_base: str = "http://localhost:11434"
    model: str = "mistral"
    # LLM is opt-in (disabled unless --llm or --llm-model is provided)
    llm: bool = False
    llm_model: str | None = None
    memdir: str | None = None
    # Voice / STT
    mic_device: int | None = None
    sample_rate: int = 16000
    whisper_model: str = "small.en"     # faster-whisper model name
    vad_aggressiveness: int = 2         # 0..3 (more aggressive = fewer false positives)
    voice: bool = False  # mic/STT input
    tts_enabled: bool = True
    tts_voice: str | None = "Zira"
    tts_rate: int = 155
    tts_volume: float = 0.9
    log_level: str = "INFO"
    ui: str = "repl"  # repl | textual

    def __post_init__(self) -> None:
        # Canonical memdir for all subsystems (logging, memory, tools, future files).
        if not self.memdir:
            self.memdir = str(resolve_memdir_cli(None))

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_SYSTEM = (
    "You are MicroBrain, a concise, technically inclined, and playful local assistant. "
    "Answer the user directly in a relaxed, natural tone. "
    "Avoid generic support phrases like 'How can I assist you today?' unless the user explicitly asks for them. "
    "Do not repeat the same sentence across turns; focus on specific, actionable replies."
)
