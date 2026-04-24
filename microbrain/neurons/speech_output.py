import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.orchestrator.debug_utils import is_debug_enabled
from microbrain.utils.memdir import resolve_memdir_ctx
from microbrain.ipc.file_inbox import IPCFileWriter

try:
    from microbrain.voice.tts import TTS  # optional; only needed for local backend
    _TTS_IMPORT_ERROR = None
except Exception as _e:
    TTS = None
    _TTS_IMPORT_ERROR = repr(_e)


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
        self._mouth_writer: IPCFileWriter | None = None
        self._mouth_memdir: Path | None = None

    async def _choose_transport(self, ctx, channel: str) -> list[str]:
        mode = str(await ctx.get_kv("speech:transport_mode", "auto") or "auto").lower()
        default_transport = str(await ctx.get_kv("speech:default_transport", "local") or "local").lower()
        audio_preferred = str(await ctx.get_kv("speech:audio_preferred_transport", "ipc") or "ipc").lower()
        interaction = await ctx.get_kv("interaction:last_input", {}) or {}

        if channel not in ("repl", "default", "cli"):
            return ["none"]

        if mode in ("ipc", "local", "none"):
            return [mode]

        now = time.time()
        spoken_bias_until = 0.0
        if isinstance(interaction, dict):
            try:
                spoken_bias_until = float(interaction.get("spoken_bias_until", 0.0) or 0.0)
            except Exception:
                spoken_bias_until = 0.0

        transports: list[str] = []
        if spoken_bias_until > now:
            transports.append(audio_preferred)
        transports.append(default_transport)

        cleaned: list[str] = []
        for item in transports:
            item = str(item or "none").lower()
            if item not in ("ipc", "local", "none"):
                item = "none"
            if item not in cleaned:
                cleaned.append(item)
        return cleaned or ["none"]

    async def _publish_ipc(self, ctx, event: Event, text: str, channel: str, style: str, voice, rate, volume, sha_text: str) -> bool:
        memdir = Path(await resolve_memdir_ctx(ctx, fallback=r"Z:\memory"))
        if self._mouth_writer is None or self._mouth_memdir != memdir:
            self._mouth_writer = IPCFileWriter(memdir=memdir, src=self.name, inbox_rel=Path("ipc/outbox"))
            self._mouth_memdir = memdir

        published = self._mouth_writer.publish(
            topic="act/speak",
            payload={
                "text": text,
                "voice": voice,
                "rate": int(rate),
                "volume": float(volume),
                "expected_sha1": sha_text,
            },
            correlation_id=event.correlation_id,
            meta={"channel": channel, "style": style, "via": "speech_output"},
        )
        await ctx.set_kv(
            "mouth:last_enqueue",
            {
                "ts": time.time(),
                "text": text,
                "queued": bool(published),
                "path": str(published) if published else "",
                "transport": "ipc",
            },
        )
        return published is not None

    async def _speak_local(self, ctx, text: str, voice, rate, volume) -> bool:
        if TTS is None:
            await ctx.set_kv(
                "tts:last_error",
                {
                    "ts": time.time(),
                    "error": f"local_tts_unavailable: {_TTS_IMPORT_ERROR}",
                    "text": text,
                },
            )
            await ctx.log_warn(
                "[speech_output] Local TTS unavailable; speech_output still loaded, but local backend cannot speak",
                import_error=_TTS_IMPORT_ERROR,
            )
            return False

        cfg_tuple = (voice, int(rate), float(volume))
        if self._tts is None or self._tts_cfg != cfg_tuple:
            self._tts = TTS(rate=int(rate), volume=float(volume), preferred=voice or "")
            self._tts_cfg = cfg_tuple
        self._tts.say(text)
        await ctx.set_kv(
            "mouth:last_enqueue",
            {
                "ts": time.time(),
                "text": text,
                "queued": True,
                "path": "local_tts",
                "transport": "local",
            },
        )
        return True

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        await ctx.set_kv(
            "speech_output:last_seen",
            {
                "ts": time.time(),
                "topic": event.topic,
                "source": event.source,
            },
        )

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

        trainer_pending = bool(await ctx.get_kv("control:t_pending", False))
        if trainer_pending and not (event.meta or {}).get("control"):
            await ctx.set_kv(
                "speech_output:last_suppressed",
                {"ts": time.time(), "text": text, "reason": "trainer_pending", "source": event.source},
            )
            return []

        # Internal thought should be stored, not spoken aloud.
        if channel == "thought":
            thought_entry = {
                "ts": time.time(),
                "text": text,
                "style": style,
                "source": event.source,
            }
            try:
                recent = await self.load_state(ctx, "recent_thoughts", default=[])
                if not isinstance(recent, list):
                    recent = []
                recent.append(thought_entry)
                if len(recent) > 16:
                    recent = recent[-16:]
                await self.save_state(ctx, "recent_thoughts", recent)
                await ctx.set_kv("thought:last", thought_entry)
            except Exception:
                pass

            if is_debug_enabled():
                print(f"[THOUGHT:{style}] {text}")
            return []

        # Always print something visible
        if is_debug_enabled():
            print(f"[SPEECH:{channel}:{style}] {text}")
        else:
            if channel == "repl":
                print(f"bot> {text}")
            else:
                print(text)

        # --- Optional TTS / Mouth sidecar ------------------------------------
        try:
            enabled = await ctx.get_kv("tts:enabled", True)
            if enabled and channel in ("repl", "default", "cli"):
                voice = await ctx.get_kv("tts:voice", None)
                rate = await ctx.get_kv("tts:rate", 155)
                volume = await ctx.get_kv("tts:volume", 0.9)

                # Mark what we are about to speak (helps detect self-echo on the mic).
                try:
                    sha_text = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
                    await ctx.set_kv("tts:last_spoken", {"ts": time.time(), "text": text, "sha1_text": sha_text})
                    await ctx.set_kv(
                        "mouth:last_intended",
                        {"ts": time.time(), "text": text, "channel": channel, "expected_sha1": sha_text},
                    )
                    # Briefly mute text-ingestion from ears to prevent response loops.
                    est_s = max(0.6, len(text) / 14.0)
                    await ctx.set_kv("ears:mute_until", time.time() + est_s + 0.25)
                except Exception:
                    sha_text = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

                selected = await self._choose_transport(ctx, channel)
                await ctx.set_kv(
                    "speech:last_transport_plan",
                    {"ts": time.time(), "text": text, "channel": channel, "plan": list(selected)},
                )

                for transport in selected:
                    if transport == "none":
                        await ctx.set_kv(
                            "speech:last_transport_choice",
                            {"ts": time.time(), "text": text, "channel": channel, "transport": "none"},
                        )
                        return []
                    if transport == "ipc":
                        ok = await self._publish_ipc(ctx, event, text, channel, style, voice, rate, volume, sha_text)
                        await ctx.set_kv(
                            "speech:last_transport_choice",
                            {"ts": time.time(), "text": text, "channel": channel, "transport": "ipc", "ok": bool(ok)},
                        )
                        if ok:
                            return []
                        continue
                    if transport == "local":
                        ok = await self._speak_local(ctx, text, voice, rate, volume)
                        await ctx.set_kv(
                            "speech:last_transport_choice",
                            {"ts": time.time(), "text": text, "channel": channel, "transport": "local", "ok": bool(ok)},
                        )
                        if ok:
                            return []
        except Exception as exc:
            try:
                await ctx.set_kv("tts:last_error", {"ts": time.time(), "error": repr(exc), "text": text})
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
