from __future__ import annotations


class MicroBrain:
    def __init__(self, ollama, mem, neuron_names: list[str]):
        self.ollama, self.mem = ollama, mem
        registry = {
            "planner": PlannerNeuron,
            "reasoner": ReasonerNeuron,
            "memory": MemoryNeuron,
            "coder": CoderNeuron,
        }
        self.neurons = [registry[n](ollama, mem) for n in neuron_names if n in registry]
        self.bus: Deque[MBMessage] = deque(maxlen=50)

    def run(self, goal: str, steps: int = 8) -> list[MBMessage]:
        self.bus.append(MBMessage(role="system", content=f"GOAL: {goal}"))
        for i in range(steps):
            for neuron in self.neurons:
                msg = neuron.step(list(self.bus), goal)
                self.bus.append(msg)
                # stop condition
                if "DONE" in msg.content.upper():
                    return list(self.bus)
        return list(self.bus)
