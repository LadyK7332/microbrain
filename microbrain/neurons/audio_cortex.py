from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator

NEURON_NAME = Path(__file__).stem


class AudioCortexNeuron(BaseNeuron):
    """
    Audio cortex adapter.

    Listens on:
        - "percept/audio"

    Expected payload shape from an external STT / audio daemon:

        {
            "text": "recognized speech here",
            "confidence": 0.92,           # optional
            "speaker": "user",            # optional (user / other / unknown)
            "channel": "repl",            # optional, defaults to "repl"
            "raw_meta": { ... }           # any extra info
        }

    Behavior:

      - Validates and normalizes the incoming audio transcription.
      - Emits a "percept/text" event so the rest of the system can
        treat it exactly like normal text input (CLI, etc.).
      - Marks the source as "mic" in raw_meta, so downstream neurons
        can distinguish spoken vs typed input if they care.
    """

    async def process(self, event: Event, ctx) -> Iterable[Event]:
        # --- debug roll call (only active when --debug is passed) ----
        self.debug(
            "received",
            topic=event.topic,
            payload=(
                {**event.payload, "pcm_bytes": f"<bytes {len(event.payload.get('pcm_bytes', b''))}>"}
                if isinstance(event.payload, dict) and 'pcm_bytes' in event.payload
                else event.payload
            ),
            source=event.source,
            meta=event.meta,
        )

        if event.topic not in ("percept/audio", "percept/audio_utterance"):
            return []

        payload = event.payload or {}
        if not isinstance(payload, dict):
            await ctx.log_warn(
                f"[{self.name}] Unexpected payload type for percept/audio",
                payload_type=str(type(payload)),
            )
            return []

        raw_text = str(payload.get("text", "") or "").strip()
        if not raw_text:
            await ctx.log_debug(
                f"[{self.name}] Empty audio transcription; nothing to forward",
                topic=event.topic,
            )
            return []

        # Only finalized utterances should become percept/text. The broader
                # percept/audio event often carries the same transcription and causes
        # duplicate forwarding into the bus.
        if event.topic == "percept/audio":
            await ctx.log_debug(
                f"[{self.name}] Ignoring percept/audio for text forwarding; utterance path owns STT ingress",
                topic=event.topic,
                text_preview=raw_text[:80],
            )
            return []

        # Suppress likely self-echo from MB's own recent speech before we
        # convert the utterance into percept/text.
        try:
            mute_until = float(await ctx.get_kv("ears:mute_until", 0.0) or 0.0)
            last_spoken = await ctx.get_kv("tts:last_spoken", {}) or {}
            now = time.time()

            if mute_until and now < mute_until:
                await ctx.log_debug(
                    f"[{self.name}] Suppressing likely self-echo during mute window",
                    text_preview=raw_text[:80],
                )
                return []

            if isinstance(last_spoken, dict):
                last_text = str(last_spoken.get("text", "") or "").strip().lower()
                last_ts = float(last_spoken.get("ts", 0.0) or 0.0)
                if last_text and last_text == raw_text.lower() and (now - last_ts) <= 8.0:
                    await ctx.log_debug(
                        f"[{self.name}] Suppressing confirmed self-echo match",
                        text_preview=raw_text[:80],
                    )
                    return []
        except Exception:
            pass

        confidence = payload.get("confidence", None)
        speaker = payload.get("speaker", "user")
        channel = str(payload.get("channel", "repl"))

        # Spoken interaction should bias MB toward spoken replies for a short
        # while, but the core reasoner should remain transport-agnostic.
        
        try:
            ttl_s = float(await ctx.get_kv("speech:audio_bias_ttl_s", 45.0) or 45.0)
            now = time.time()
            await ctx.set_kv(
                "interaction:last_input",
                {
                    "ts": now,
                    "source": "user",
                    "transport_source": "mic",
                    "channel": channel,
                    "modality": "audio",
                    "text": raw_text[:160],
                    "spoken_bias_until": now + max(0.0, ttl_s),
                },
            )
        except Exception:
            pass

        raw_meta: Dict[str, Any] = payload.get("raw_meta", {}) or {}
        # Spoken input is semantically from the user, transported by mic/audio.
        raw_meta.update(
            {
                "input_modality": "audio",
                "speaker": speaker,
                "stt_confidence": confidence,
                "transport_source": "mic",
                "source": "user",
                "channel": channel,
                "sensor_lobe": "audio",
                "adapter": self.name,
            }
        )

        # Build normalized text percept payload
        text_payload: Dict[str, Any] = {
            "text": raw_text,
            "source": "user",
            "channel": channel,
            "raw_meta": raw_meta,
        }

        await ctx.log_debug(
            f"[{self.name}] Forwarding audio transcription as text",
            channel=channel,
            speaker=speaker,
            confidence=confidence,
            text_preview=raw_text[:80],
        )

        text_event = Event(
            topic="percept/text",
            payload=text_payload,
            source=self.name,
            correlation_id=event.correlation_id,
        )

        return [text_event]


def build_neurons(orchestrator: Orchestrator):
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=["percept/audio", "percept/audio_utterance"],
        output_topics=["percept/text"],
        # Priority: early, so text router / attention see it quickly.
        priority=1,
    )
    yield AudioCortexNeuron(cfg)
