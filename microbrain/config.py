from dataclasses import asdict, dataclass


@dataclass
class AppConfig:
    ollama_base: str = "http://localhost:11434"
    model: str = "mistral"
    onnx_embed_path: str | None = None
    onnx_provider: str | None = None
    onnx_max_len: int = 256
    memdir: str | None = None
    vosk_model_path: str | None = None
    mic_device: int | None = None
    sample_rate: int = 16000
    tts_voice: str | None = None
    tts_rate: int = 170
    tts_volume: float = 1.0
    log_level: str = "INFO"

    def as_dict(self) -> dict:
        return asdict(self)
