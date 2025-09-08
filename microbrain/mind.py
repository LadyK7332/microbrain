from __future__ import annotations
import argparse, asyncio
from microbrain.config import AppConfig
from microbrain.utils.logging_setup import configure_logging

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

async def main_async(cfg: AppConfig):
    logger = configure_logging(cfg.log_level)
    from microbrain.ollama_client import OllamaClient
    from microbrain.memory.memory_store import MemoryStore
    from microbrain.tools import ToolRegistry
    from microbrain.agent import Agent

    client = OllamaClient(host=cfg.ollama_base, model=cfg.model)
    mem = MemoryStore(memdir=cfg.memdir, onnx_embed_path=cfg.onnx_embed_path, onnx_provider=cfg.onnx_provider, onnx_max_len=cfg.onnx_max_len)
    tools = ToolRegistry()
    agent = Agent(client=client, memory=mem, tools=tools, logger=logger)

if cfg.voice:
    if not cfg.vosk_model_path:
        logger.error("Voice requested but --vosk-model-path is missing.")
        return
    from microbrain.voice.asr import STT
    from microbrain.voice.tts import TTS
    from microbrain.voice.loop import voice_chat_loop
    stt = STT(model_path=cfg.vosk_model_path, mic_device=cfg.mic_device,
              sample_rate=cfg.sample_rate, logger=logger)
    tts = TTS(voice_name=cfg.tts_voice, rate=cfg.tts_rate,
              volume=cfg.tts_volume, logger=logger)
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
