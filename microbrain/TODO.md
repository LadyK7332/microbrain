## Config precedence (MicroBrain)

Effective order for key settings:

1. **CLI flags** (highest)
   - `--model`              → override model path/name
   - `--memdir`             → override memory directory
   - `--llama-backend` / `--vulkan`
   - `--voice`, `--whisper-model-path`, `--mic-device`
   - `--tts-voice`, `--tts-rate`, `--tts-volume`
   - `--log-level`

2. **Environment variables**
   - `MB_LLAMA_MODEL`       → default model path if `--model` not given
   - `MB_MEMDIR`            → default memdir if `--memdir` not given

3. **AppConfig defaults** (in `microbrain/config.py`)
   - `model`                → fallback model
   - `memdir`               → fallback memdir
   - ONNX / voice / misc defaults

Example:

```powershell
$env:MB_LLAMA_MODEL = "C:\aiproj\microbrain\microbrain\models\DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf"
$env:MB_MEMDIR      = "Z:\memory"

python -m microbrain.mind --vulkan --log-level INFO
