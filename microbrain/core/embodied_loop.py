import asyncio
import time

from utils.metrics import push_metrics

from microbrain.core.reflex_gate import ReflexGate
from microbrain.core.reflex_trace import ReflexTrace
from microbrain.core.synaptic_loop import SynapticLoop


class EmbodiedLoop:
    def __init__(self):
        self.gate = ReflexGate()
        self.syn = SynapticLoop()
        self.trace = ReflexTrace()
        self.hits = 0
        self.misses = 0
        self.heartbeat = 0.5  # seconds between ticks

    async def tick(self, senses):
        score = self.syn.score(senses)
        if self.gate.decide(score):
            self.hits += 1
            decision = "interrupt"
            useful = True
            self.syn.update(senses, True)
        else:
            self.misses += 1
            decision = "route"
            useful = False
            self.syn.update(senses, False)
        self.trace.record(senses, score, decision, useful)
        self.push_metrics()
        await asyncio.sleep(self.heartbeat)

    def push_metrics(self):
        summary = self.trace.summarize()
        push_metrics(
            threshold=self.gate.theta,
            hits=summary["hits"],
            misses=summary["misses"],
            workers={},
            notes=f"heartbeat@{time.time():.0f}",
        )
