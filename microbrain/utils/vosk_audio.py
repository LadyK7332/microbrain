from __future__ import annotations

import json
import queue
import threading
from typing import Callable, Optional

import sounddevice as sd
from vosk import Model, KaldiRecognizer


class VoskAudioListener:
    """
    Simple Vosk-based microphone listener.

    - Uses default input device (system mic).
    - 16 kHz mono stream.
    - On each *final* STT result, calls the provided callback(text: str).
    """

    def __init__(
        self,
        model_path: str,
        on_transcript: Callable[[str], None],
        samplerate: int = 16000,
        blocksize: int = 8000,
    ) -> None:
        self.model_path = model_path
        self.on_transcript = on_transcript
        self.samplerate = samplerate
        self.blocksize = blocksize

        self._model: Optional[Model] = None
        self._recognizer: Optional[KaldiRecognizer] = None
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._stream: Optional[sd.RawInputStream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # You can print status if you want
            # print(status, flush=True)
            pass
        # indata is bytes in RawInputStream
        self._queue.put(bytes(indata))

    def start(self) -> None:
        if self._thread is not None:
            return  # already running

        # Load model
        self._model = Model(self.model_path)
        self._recognizer = KaldiRecognizer(self._model, self.samplerate)
        self._recognizer.SetWords(True)

        self._stop_flag.clear()

        # Start audio stream
        self._stream = sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=1,
            device=24
            100000000000000000,  # Logitech G432 mic 12 or 6
            callback=self._audio_callback,
        )
        self._stream.start()

        # Start processing thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        assert self._recognizer is not None
        while not self._stop_flag.is_set():
            try:
                data = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._recognizer.AcceptWaveform(data):
                result = self._recognizer.Result()
                try:
                    obj = json.loads(result)
                except Exception:
                    obj = {}
                text = (obj.get("text") or "").strip()
                if text:
                    # Call callback with the recognized text
                    self.on_transcript(text)
            else:
                # Partial result (can be used if you want realtime)
                # partial = json.loads(self._recognizer.PartialResult()).get("partial", "")
                # we ignore partial for now
                pass

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
