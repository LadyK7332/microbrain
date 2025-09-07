from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from typing import Optional

@dataclass
class AppConfig:
    ollama_base: str = "http://localhost:11434"
    model: str = "mistral"
    onnx_embed_path: Optional[str] = None
    onnx_provider: Optional[str] = None
    onnx_max_len: int = 256
    memdir: Optional[str] = None
    vosk_model_path: Optional[str] = None
    mic_device: Optional[int] = None
    sample_rate: int = 16000
    tts_voice: Optional[str] = None
    tts_rate: int = 170
    tts_volume: float = 1.0
    log_level: str = "INFO"

    def as_dict(self) -> dict:
        return asdict(self)
