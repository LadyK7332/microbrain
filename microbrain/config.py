from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class AppConfig:
    onnx_embed_path: str | None = None
    onnx_provider: str | None = None
    onnx_max_len: int = 256
    ollama_base: str = "http://localhost:11434"
    model: str = "mistral"
    memdir: str | None = None
    # Voice / STT
    mic_device: int | None = None
    sample_rate: int = 16000
    whisper_model: str = "small.en"     # faster-whisper model name
    vad_aggressiveness: int = 2         # 0..3 (more aggressive = fewer false positives)
    voice: bool = False
    tts_voice: str | None = None
    tts_rate: int = 170
    tts_volume: float = 1.0
    log_level: str = "INFO"
    voice: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_SYSTEM = (
    "You are MicroBrain, a concise, technically inclined local assistant. "
    "Answer the user directly in a relaxed, natural tone. "
    "Avoid generic support phrases like 'How can I assist you today?' unless the user explicitly asks for them. "
    "Do not repeat the same sentence across turns; focus on specific, actionable replies."
)
