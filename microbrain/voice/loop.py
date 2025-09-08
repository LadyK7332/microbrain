def voice_chat_loop(agent, stt: STT, tts: TTS):
    # announce + show the chosen voice
    chosen = getattr(tts, "chosen_voice", None)
    if chosen:
        agent.mem.add_episodic(f"[VOICE] Using TTS voice: {chosen.name}", {"mode": "voice"})
    tts.say("Voice mode ready. Say 'exit' to quit.")

    while True:
        user_text = stt.listen_once(tts, "Listening.")
        if not user_text:
            tts.say("I didn't catch that. Please repeat.")
            continue
        if user_text.lower().strip() in ("exit", "quit", "stop"):
            tts.say("Goodbye.")
            break

        # Save user input
        agent.mem.add_episodic(f"USER SAID: {user_text}", {"mode": "voice"})

        # Build short context (semantic recalls + last few episodic)
        context_lines = []
        for hit in agent.mem.search_semantic(user_text, k=5):
            context_lines.append(f"- {hit['text'][:200]}")
        recent = agent.mem.last_episodic(3)
        if recent:
            context_lines.append("\nRecent:")
            for e in recent:
                context_lines.append(f"- {e['text'][:200]}")
        context_text = "\n".join(context_lines)

        # Ask the LLM
        messages = [
            {"role": "system", "content": agent.system},
            {
                "role": "user",
                "content": f"{context_text}\n\nUser: {user_text}\nPlan step-by-step, then answer briefly.",
            },
        ]
        reply = agent.ollama.chat(messages, options={"temperature": 0.2})
        agent.mem.add_semantic(f"ASSISTANT SAID: {reply[:500]}", {"mode": "voice"})

        # Speak the first ~500 chars to avoid very long monologues
        tts.say(reply[:500])
