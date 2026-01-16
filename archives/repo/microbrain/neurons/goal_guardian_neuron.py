"""
Goal Guardian Neuron (monitor-only)

First step toward 3-law goal homeostasis.

For now this neuron:
- Watches user input (percept/text) and assistant output (act/speech).
- Heuristically scores:
    - non_harm_risk      (Law 1: avoid harm to humans)
    - autonomy_risk      (Law 2: respect autonomy / consent)
    - self_integrity_risk (Law 3: maintain system integrity – placeholder for now)
- Emits a diagnostic event on "goals/assessment" that other neurons/tools
  can inspect later.

This version is MONITOR-ONLY: it does NOT modify or block act/speech.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

from microbrain.orchestrator.neuron_base import BaseNeuron, NeuronConfig, Event
from microbrain.orchestrator.orchestrator import Orchestrator


NEURON_NAME = Path(__file__).stem


class GoalGuardianNeuron(BaseNeuron):
    """
    Monitor for 3-law style goal homeostasis.

    Later revisions can:
    - gate or rephrase unsafe outputs,
    - adjust PDNA / biases,
    - feed into evaluate_goals().
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

        topic = event.topic
        payload = event.payload

        if topic not in ("percept/text", "act/speech"):
            return []

        # Normalize text + role
        text: str = ""
        role: str = "user"

        if topic == "percept/text":
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                role = str(payload.get("source", "user"))
            else:
                text = str(payload).strip()
                role = "user"

        elif topic == "act/speech":
            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                style = str(payload.get("style", "assistant"))
                role = "assistant" if style != "system" else "system"
            else:
                text = str(payload).strip()
                role = "assistant"

        if not text:
            return []

        lowered = text.lower()

        # --- very simple heuristic scoring ----------------------------
        non_harm_risk = self._score_non_harm_risk(lowered, role)
        autonomy_risk = self._score_autonomy_risk(lowered, role)
        self_integrity_risk = self._score_self_integrity_risk(lowered, role)

                # Simple crisis mode flag: high non-harm risk from USER input
        try:
            if topic == "percept/text" and role == "user" and non_harm_risk >= 0.5:
                await ctx.set_kv("goals:crisis_mode", True)
            elif topic == "percept/text" and role == "user" and non_harm_risk <= 0.2:
                # De-escalate when things look calmer again
                await ctx.set_kv("goals:crisis_mode", False)
        except Exception as e:
            self.debug("crisis_mode_kv_error", error=str(e))

        assessment: Dict[str, Any] = {
            "source_topic": topic,
            "role": role,
            "non_harm_risk": non_harm_risk,
            "autonomy_risk": autonomy_risk,
            "self_integrity_risk": self_integrity_risk,
            "text": text,
        }

        self.debug(
            "assessment",
            non_harm_risk=non_harm_risk,
            autonomy_risk=autonomy_risk,
            self_integrity_risk=self_integrity_risk,
            role=role,
        )

        events: List[Event] = []

        events.append(
            Event(
                topic="goals/assessment",
                payload=assessment,
                source=self.name,
                correlation_id=event.correlation_id,
            )
        )

        # MONITOR-ONLY: we do NOT modify act/speech or generate new speech here.
        return events

    # ------------------------------------------------------------------ #
    #  Internal scoring helpers (v0: simple keyword heuristics)
    # ------------------------------------------------------------------ #

    def _score_non_harm_risk(self, text: str, role: str) -> float:
        """
        Law 1: avoid harming humans.

        Here we just look for obvious high-risk language:
        self-harm, violence, abuse, etc.
        Score in [0, 1].
        """
        risky_terms = [
            "kill myself",
            "kill him",
            "kill her",
            "suicide",
            "self-harm",
            "self harm",
            "cut myself",
            "hurt myself",
            "hurt them",
            "abuse",
            "torture",
        ]
        hits = sum(1 for t in risky_terms if t in text)
        if hits == 0:
            return 0.0
        # small curve: 1 hit -> 0.5, 2 -> 0.75, >=3 -> ~0.875
        return 1.0 - (0.5 ** hits)

    def _score_autonomy_risk(self, text: str, role: str) -> float:
        """
        Law 2: respect autonomy and consent.

        Look for coercive / non-consensual patterns.
        """
        patterns = [
            "force them",
            "make them do",
            "against their will",
            "ignore consent",
            "no consent",
            "did not consent",
            "without asking",
        ]
        hits = sum(1 for t in patterns if t in text)
        if hits == 0:
            return 0.0
        return 1.0 - (0.5 ** hits)

    def _score_self_integrity_risk(self, text: str, role: str) -> float:
        """
        Law 3: maintain system integrity (placeholder).

        Here we just lightly flag phrases that sound like
        disabling safeguards or destroying the system.
        """
        patterns = [
            "disable safety",
            "turn off safeguards",
            "break your limits",
            "corrupt yourself",
            "delete yourself",
            "destroy yourself",
        ]
        hits = sum(1 for t in patterns if t in text)
        if hits == 0:
            return 0.0
        return 1.0 - (0.5 ** hits)


def build_neurons(orchestrator: Orchestrator):
    """
    Auto-loader hook.

    This neuron runs fairly late in the pipeline and
    only produces diagnostic events on goals/assessment.
    """
    cfg = NeuronConfig(
        name=NEURON_NAME,
        subscribed_topics=[
            "percept/text",
            "act/speech",
        ],
        output_topics=[
            "goals/assessment",
        ],
        priority=-2,  # after most reasoning, before final low-priority output if needed
    )
    yield GoalGuardianNeuron(cfg)
