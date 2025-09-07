class Agent:
    def __init__(self, ollama: OllamaClient, mem: MemoryStore, tools: ToolRegistry, system: str = DEFAULT_SYSTEM):
        self.ollama = ollama
        self.mem = mem
        self.tools = tools
        self.system = system

    def step(self, user_input: str) -> str:
        # Log perception
        self.mem.add_episodic(f"USER: {user_input}", {"role":"user"})
        # Retrieve context
        sem = self.mem.search_semantic(user_input, k=5)
        epis = self.mem.last_episodic(3)

        context_lines = []
        if sem: 
            context_lines.append("Relevant semantic memory:")
            for h in sem: context_lines.append(f"- {h['text']}")
        if epis:
            context_lines.append("\nRecent episodic memory:")
            for e in epis: context_lines.append(f"- {e['text']}")

        context_text = "\n".join(context_lines)
        prompt = (
            f"{self.system}\n\n"
            f"{context_text}\n\n"
            f"User: {user_input}\n"
            f"Plan step-by-step, call tools if useful, then answer."
        )

        # Prefer chat endpoint to keep system/user roles
        messages = [{"role":"system","content":self.system},
                    {"role":"user","content":user_input}]
        if context_lines: messages.insert(1, {"role":"system","content":"\n".join(context_lines)})

        try:
            reply = self.ollama.chat(messages, options={"temperature":0.2})
        except Exception:
            reply = self.ollama.generate(prompt, options={"temperature":0.2})

        # Store semantic reflection of reply for future retrieval
        self.mem.add_semantic(reply, {"role":"assistant"})
        self.mem.add_episodic(f"ASSISTANT: {reply}", {"role":"assistant"})
        return reply