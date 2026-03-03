import hashlib
import time
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.orchestrator.debug_utils import is_debug_enabled
from microbrain.voice.tts import TTS



class SpeechOutputNeuron(BaseNeuron):
    """
    Terminal sink that prints speech actions to the console.

    - In normal mode: prints only the assistant reply as `bot> ...`.
    - In debug mode: prints detailed `[SPEECH:channel:style] ...` lines.
    """

    def __init__(self, config: NeuronConfig):
        super().__init__(config)
        self._tts: TTS | None = None
        self._tts_cfg: tuple[str | None, int, float] | None = None

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=event.payload,
            source=event.source,
            meta=event.meta,
        )

        payload = event.payload
        channel = "cli"
        style = "default"
        text = ""

        # --- Normalize payload ------------------------------------------------
        if isinstance(payload, str):
            text = payload
        elif isinstance(payload, dict):
            text = str(payload.get("text", ""))
            channel = str(payload.get("channel", channel))
            style = str(payload.get("style", style))
        else:
            text = str(payload)

        text = text.strip()
        if not text:
            return []

        # Always print something visible
        if is_debug_enabled():
            print(f"[SPEECH:{channel}:{style}] {text}")
        else:
            if channel == "repl":
                print(f"bot> {text}")
            else:
                print(text)

        # --- Optional TTS -----------------------------------------------------
        try:
            enabled = await ctx.get_kv("tts:enabled", True)
            if enabled and channel in ("repl", "default", "cli"):
                voice = await ctx.get_kv("tts:voice", None)
                rate = await ctx.get_kv("tts:rate", 155)
                volume = await ctx.get_kv("tts:volume", 0.9)

                cfg_tuple = (voice, int(rate), float(volume))
                if self._tts is None or self._tts_cfg != cfg_tuple:
                    self._tts = TTS(rate=int(rate), volume=float(volume), preferred=voice or "")
                    self._tts_cfg = cfg_tuple

                # Mark what we are about to speak (helps detect self-echo on the mic).
                try:
                    sha_text = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                    await ctx.set_kv("tts:last_spoken", {"ts": time.time(), "text": text, "sha1_text": sha_text})
                    # Briefly mute text-ingestion from ears to prevent response loops.
                    est_s = max(0.6, len(text) / 14.0)
                    await ctx.set_kv("ears:mute_until", time.time() + est_s + 0.25)
                except Exception:
                    pass

                self._tts.say(text)
        except Exception as exc:
            try:
                await ctx.log_warn(f"[speech_output] TTS error: {exc!r}", topic=event.topic)
            except Exception:
                pass
            if is_debug_enabled():
                print(f"[SPEECH:TTS_ERROR] {exc!r}")

        # This is a sink: no further events
        return []


def build_neurons(orchestrator: Orchestrator):
    """
    Auto-loader hook.

    This will be picked up by:
        microbrain.orchestrator.neuron_loader.auto_register_neurons(...)
    """
    cfg = NeuronConfig(
        name="speech_output",
        subscribed_topics=["act/speech"],
        output_topics=[],  # terminal sink
        priority=-10,      # usually runs after "thinking" neurons
    )
    yield SpeechOutputNeuron(cfg)
