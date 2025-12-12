from typing import Any, Dict, Iterable

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator
from microbrain.orchestrator.debug_utils import is_debug_enabled



class SpeechOutputNeuron(BaseNeuron):
    """
    Terminal sink that prints speech actions to the console.

    - In normal mode: prints only the assistant reply as `bot> ...`.
    - In debug mode: prints detailed `[SPEECH:channel:style] ...` lines.
    """

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
            # Nothing meaningful to say, just drop it
            return []

        # --- Debug vs normal behavior ----------------------------------------
        if is_debug_enabled():
            # Dev mode: keep detailed SPEECH lines
            print(f"[SPEECH:{channel}:{style}] {text}")
        else:
            # Normal mode: only show assistant replies as `bot>`
            # (LLMReasoner emits channel='repl', style='assistant')
            if channel == "repl":
                print(f"bot> {text}")
            # Echo or other channels are suppressed in normal mode

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
