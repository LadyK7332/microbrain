class STT:
    """
    Vosk STT with sounddevice. Uses device default samplerate unless you pass one.
    """

    def __init__(self, vosk_model_path: str, device: int | None = None, samplerate: int = 0):
        if not vosk_model_path or not os.path.isdir(vosk_model_path):
            raise RuntimeError("Invalid --vosk-model-path (folder not found).")

        # Resolve device + samplerate
        self.device = device
        if device is not None:
            info = sd.query_devices(device)
            if not samplerate or samplerate <= 0:
                samplerate = int(info["default_samplerate"])
        else:
            # fallback to default input device’s rate
            info = sd.query_devices(None, "input")
            if not samplerate or samplerate <= 0:
                samplerate = int(info["default_samplerate"])

        self.rate = samplerate
        self.model = VoskModel(vosk_model_path)
        self.rec = KaldiRecognizer(self.model, self.rate)
        self.rec.SetWords(True)
        self._q: queue.Queue[bytes] = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:
            # non-fatal stream status (overruns etc.)
            pass
        self._q.put(bytes(indata))

    def listen_once(self, prompt_tts: "TTS | None" = None, prompt_text: str = "Listening.") -> str:
        if prompt_tts:
            prompt_tts.say(prompt_text)

        # Use explicit device if provided
        kwargs = dict(
            samplerate=self.rate, blocksize=4096, dtype="int16", channels=1, callback=self._callback
        )
        if self.device is not None:
            kwargs["device"] = self.device

        # Collect ~5 seconds or until a final result
        import time as _t

        t0 = _t.time()
        with sd.RawInputStream(**kwargs):
            partial = ""
            while True:
                data = self._q.get()
                if self.rec.AcceptWaveform(data):
                    import json as _json

                    res = _json.loads(self.rec.Result())
                    text = (res.get("text") or "").strip()
                    if text:
                        return text
                else:
                    import json as _json

                    p = _json.loads(self.rec.PartialResult()).get("partial", "")
                    if p:
                        partial = p
                if _t.time() - t0 > 5:
                    return partial.strip()
