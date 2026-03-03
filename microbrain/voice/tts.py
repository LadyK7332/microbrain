import pyttsx3


class TTS:
    """
    Offline TTS using Windows SAPI via pyttsx3.
    You can pass a name substring (e.g., 'Zira', 'Aria', 'Female') to pick a mature feminine voice.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0, preferred: str = ""):
        self.engine = pyttsx3.init()
        # rate / volume
        try:
            self.engine.setProperty("rate", rate)
        except Exception:
            pass
        try:
            self.engine.setProperty("volume", max(0.0, min(1.0, volume)))
        except Exception:
            pass

        # voice selection by substring (case-insensitive)
        self.chosen_voice = None
        try:
            voices = self.engine.getProperty("voices") or []
            pref = (preferred or "").lower()
            if pref:
                for v in voices:
                    name = (getattr(v, "name", "") or "").lower()
                    if pref in name:
                        self.engine.setProperty("voice", v.id)
                        self.chosen_voice = v
                        break

            # If no match and we want a feminine voice, try common female voices by name
            if not self.chosen_voice and pref:
                for tag in ("female", "zira", "aria", "susan", "eva", "hazel"):
                    for v in voices:
                        if tag in (getattr(v, "name", "") or "").lower():
                            self.engine.setProperty("voice", v.id)
                            self.chosen_voice = v
                            break
                    if self.chosen_voice:
                        break
        except Exception:
            pass

    def list_voices(self) -> list[str]:
        try:
            return [
                f"{i}: {v.name} ({v.id})"
                for i, v in enumerate(self.engine.getProperty("voices") or [])
            ]
        except Exception:
            return []

    def save_to_file(self, text: str, out_path: str) -> None:
        """Save synthesized speech to a WAV file. (Does not play it.)"""
        self.engine.save_to_file(text, out_path)
        self.engine.runAndWait()

    def say(self, text: str, auto_run: bool = True):
        self.engine.say(text)
        if auto_run:
            self.engine.runAndWait()

    def runAndWait(self):
        self.engine.runAndWait()

    def stop(self):
        try:
            self.engine.stop()
        except Exception:
            pass
