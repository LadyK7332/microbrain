import pyttsx3

PREFERRED = "Zira"   # try "Aria", "Jenny", "Hazel", etc.

def main():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices") or []

    chosen = None
    for v in voices:
        name = (getattr(v, "name", "") or "")
        if PREFERRED.lower() in name.lower():
            chosen = v
            break

    if chosen:
        engine.setProperty("voice", chosen.id)
        print("Using:", chosen.name)
    else:
        print("No match for:", PREFERRED)
        print("Using default voice.")

    engine.setProperty("rate", 155)
    engine.setProperty("volume", 0.9)

    engine.say("MicroBrain mouth online. I will speak only what I mean.")
    engine.runAndWait()

if __name__ == "__main__":
    main()