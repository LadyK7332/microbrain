from microbrain.core.reflex_trace import ReflexTrace

if __name__ == "__main__":
    # Minimal demo harness, kept behind guard to avoid undefined names at import time.
    def fuse_signals(sig: dict) -> float:
        # Example scoring from a simple sensor dict
        return float(sig.get("energy", 0.0))

    class _Gate:
        def __init__(self, threshold: float = 0.6) -> None:
            self.threshold = threshold

        def decide(self, score: float) -> bool:
            return score >= self.threshold

    senses = {"energy": 0.8}
    gate = _Gate(0.6)
    score = fuse_signals(senses)
    decision = "interrupt" if gate.decide(score) else "pass"
    useful = decision == "interrupt"

    trace = ReflexTrace(capacity=10, path="reflex_trace.jsonl")
    trace.record(signal=senses, score=score, decision=decision, useful=useful)
