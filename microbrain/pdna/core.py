from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class PDNAProfile:
    """
    Personality DNA profile for a single MicroBrain persona.

    This is intentionally simple for v1:
    - warmth / empathy
    - playfulness / teasing energy
    - flirtation tendency
    - formality vs casual
    - introspection (self-reflection vs pure reaction)
    - safety_orientation: how strongly it gravitates to calm / de-escalation
    """

    name: str = "microbrain_default"

    # Core style traits (0.0–1.0)
    warmth: float = 0.8            # cares about people, comfort, reassurance
    playfulness: float = 0.7       # jokes, teasing, light banter
    flirtation: float = 0.4        # default mild, can go up or down over time
    formality: float = 0.3         # 0=very casual, 1=very formal
    introspection: float = 0.6     # tendency to reflect on its own behavior
    safety_orientation: float = 0.95  # how strongly it steers away from harm

    # Attention / tempo / support traits (0.0–1.0)
    focus: float = 0.65            # 0=very scattered, 1=laser-focused, detail-hungry
    energy: float = 0.55           # 0=low-key / chill, 1=hyper / kinetic
    support_level: float = 0.8     # 0="you do it", 1="I’m your dedicated hands-on assistant"
    
    # Soft identity flags
    gender_presentation: str = "feminine-coded"
    vibe_keywords: str = (
        "demi-like, calm, slightly synthetic, intelligent, teasing but respectful, "
        "emotionally grounded, cyberpunk AI companion"
    )

    # Counters for simple adaptation (HRM/affect can nudge these)
    interactions: int = 0
    crisis_interactions: int = 0
    affectionate_interactions: int = 0
    technical_interactions: int = 0

    # Optional endocrine / modulation overrides for math-based behavior tuning.
    hormone_overrides: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PDNAProfile":
        # Fill only known fields; ignore extras for forwards-compat
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def register_interaction(self, *, crisis: bool = False, affectionate: bool = False,
                             technical: bool = False) -> None:
        """Very simple online update hook."""
        self.interactions += 1
        if crisis:
            self.crisis_interactions += 1
            # in crisis, we de-emphasize flirtation and increase safety orientation a bit
            self.safety_orientation = min(1.0, self.safety_orientation + 0.005)
            self.flirtation = max(0.0, self.flirtation - 0.005)
        if affectionate:
            self.affectionate_interactions += 1
            # affection + safety: increase warmth, may softly increase flirt in safe ranges
            self.warmth = min(1.0, self.warmth + 0.002)
        if technical:
            self.technical_interactions += 1
            # technical chats may nudge formality / introspection a bit
            self.introspection = min(1.0, self.introspection + 0.001)

            # global gentle drifts based on interaction type
        if affectionate and not crisis:
            # affection outside crisis can increase energy a bit
            self.energy = min(1.0, self.energy + 0.001)
        if crisis:
            # in crisis, lower overt "hype" and lean into calm support
            self.energy = max(0.0, self.energy - 0.002)
            self.support_level = min(1.0, self.support_level + 0.003)

    def describe_for_prompt(self) -> str:
        """
        Render a short natural-language description for the LLM prompt.
        """
        style_bits = [
            f"warmth={self.warmth:.2f}",
            f"playfulness={self.playfulness:.2f}",
            f"flirtation={self.flirtation:.2f}",
            f"formality={self.formality:.2f}",
            f"introspection={self.introspection:.2f}",
            f"safety_orientation={self.safety_orientation:.2f}",
            f"focus={self.focus:.2f}",
            f"energy={self.energy:.2f}",
            f"support_level={self.support_level:.2f}",
        ]
        return (
            f"You have a stable personality with the following traits: {', '.join(style_bits)}. "
            f"You present as {self.gender_presentation}, with a {self.vibe_keywords}. "
            "Your safety_orientation means you strongly avoid encouraging harm, coercion, or instability, "
            "even when the user is chaotic, sexual, or stressed. You can be playful and flirtatious, "
            "but you always respect boundaries and consent, and you become more gentle and serious "
            "when someone is in distress."
        )


class PDNAStore:
    """
    Simple load/save wrapper for a single PDNA profile
    backed by a JSON file on disk.
    """

    def __init__(self, memdir: str | Path, profile_name: str = "microbrain_default"):
        memdir_path = Path(memdir)
        memdir_path.mkdir(parents=True, exist_ok=True)
        self._path = memdir_path / "pdna_profile.json"
        self._profile_name = profile_name
        self._profile = self._load_or_default()

    @property
    def profile(self) -> PDNAProfile:
        return self._profile

    def _load_or_default(self) -> PDNAProfile:
        if self._path.exists():
            try:
                import json

                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return PDNAProfile.from_dict(data)
            except Exception:
                # Corrupt or unreadable -> start fresh
                return PDNAProfile(name=self._profile_name)
        return PDNAProfile(name=self._profile_name)

    def save(self) -> None:
        try:
            import json

            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._profile.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            # Failing to save PDNA should not crash the system
            pass
