from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, List

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel


@dataclass
class WhisperAudioConfig:
    model_name: str = "small.en"
    device_index: Optional[int] = None

    # Target SR for VAD + Whisper (keep 16000 for best compatibility)
    sample_rate: int = 16000

    # If set, we will try to open the device at this SR and resample -> sample_rate.
    # If None, we try sample_rate first and then auto-fallback.
    device_sample_rate: Optional[int] = None
    fallback_device_sample_rates: tuple[int, ...] = (44100, 48000)

    # If True, bypass VAD/Whisper and just forward frames via on_audio_raw.
    raw_only: bool = False

    frame_ms: int = 30  # 10/20/30 supported by webrtcvad
    vad_aggressiveness: int = 2
    start_trigger_frames: int = 5
    end_silence_frames: int = 12
    max_utterance_seconds: float = 12.0

class CochlearResampler:
    """
    Simple streaming resampler: device_sr -> target_sr.
    Uses linear interpolation (fast, dependency-free).
    Produces int16 bytes at target_sr (mono).
    """

    def __init__(self, device_sr: int, target_sr: int) -> None:
        self.device_sr = int(device_sr)
        self.target_sr = int(target_sr)
        self._carry = np.zeros((0,), dtype=np.float32)

    def push_int16_bytes(self, chunk: bytes) -> bytes:
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if self._carry.size:
            x = np.concatenate([self._carry, x], axis=0)
            self._carry = np.zeros((0,), dtype=np.float32)

        if self.device_sr == self.target_sr:
            y = x
        else:
            # Linear resample
            n_in = x.shape[0]
            n_out = int(round(n_in * (self.target_sr / self.device_sr)))
            if n_out <= 1:
                # Save as carry and return nothing
                self._carry = x
                return b""
            t_in = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float32)
            t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
            y = np.interp(t_out, t_in, x).astype(np.float32)

        # Keep a tiny tail to stabilize next interp window
        if y.shape[0] > 0:
            # carry ~5ms of device_sr audio (roughly)
            carry_n = max(0, int(self.target_sr * 0.005))
            if y.shape[0] > carry_n:
                tail = y[-carry_n:]
                # tail is already target_sr-domain; re-use as carry in that domain
                # (this is "good enough" for linear streaming)
                self._carry = np.array([], dtype=np.float32)
                y2 = y[:-carry_n]
                y = np.concatenate([y2, tail], axis=0)

        y16 = np.clip(y, -1.0, 1.0)
        y16 = (y16 * 32767.0).astype(np.int16)
        return y16.tobytes()


class WhisperAudioListener:
    """
    Mic -> VAD -> (utterance) -> faster-whisper -> on_transcript(text)

    Runs capture in a callback thread and processing in a daemon thread.
    """

    def __init__(
        self,
        cfg: WhisperAudioConfig,
        on_transcript: Callable[[str], None],
        on_debug: Optional[Callable[[str], None]] = None,
        on_audio_raw: Optional[Callable[[bytes, int], None]] = None,
        on_utterance: Optional[Callable[[str, bytes, int], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.on_transcript = on_transcript
        self.on_debug = on_debug
        self.on_audio_raw = on_audio_raw
        self.on_utterance = on_utterance

        # capture rate = device/native rate if provided; otherwise fall back
        self._capture_rate = int(self.cfg.device_sample_rate or self.cfg.sample_rate)

        self._q: "queue.Queue[bytes]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.RawInputStream] = None

        self._vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)

        # Load whisper model once
        # NOTE: CPU by default; you can later pass device="cuda" if you add that support.
        self._model = WhisperModel(self.cfg.model_name, device="cpu", compute_type="int8")

        # Derived sizes (VAD framing is ALWAYS in target sample_rate domain)
        self._frame_samples = int(self.cfg.sample_rate * (self.cfg.frame_ms / 1000.0))
        self._frame_bytes = self._frame_samples * 2  # int16 mono @ target sr

        # Device SR is decided in start(); cochlear is created there.
        self._device_sr: Optional[int] = None
        self._cochlear: Optional[CochlearResampler] = None

    def _dbg(self, msg: str) -> None:
        if self.on_debug:
            try:
                self.on_debug(msg)
            except Exception:
                pass

    def _cb(self, indata, frames, time_info, status):
        if status:
            self._dbg(f"[whisper_audio] stream status: {status}")
        self._q.put(bytes(indata))

    def start(self) -> None:
        if self._thread is not None:
            return

        kwargs = {}
        if self.cfg.device_index is not None:
            kwargs["device"] = self.cfg.device_index

        # Decide what SR to open the device at
        candidates: List[int] = []
        if self.cfg.device_sample_rate is not None:
            candidates.append(int(self.cfg.device_sample_rate))
        candidates.append(int(self.cfg.sample_rate))
        for sr in self.cfg.fallback_device_sample_rates:
            if int(sr) not in candidates:
                candidates.append(int(sr))

        self._stop.clear()

        last_err: Optional[Exception] = None
        opened = False
        for dev_sr in candidates:
            try:
                blocksize = int(dev_sr * (self.cfg.frame_ms / 1000.0))
                self._stream = sd.RawInputStream(
                    samplerate=dev_sr,
                    dtype="int16",
                    channels=1,
                    blocksize=blocksize,
                    callback=self._cb,
                    **kwargs,
                )
                self._stream.start()
                self._device_sr = dev_sr
                opened = True
                break
            except Exception as e:
                last_err = e
                self._dbg(f"[whisper_audio] failed to open mic at {dev_sr} Hz: {e!r}")

        if not opened or self._stream is None or self._device_sr is None:
            raise RuntimeError(f"Could not open microphone at any sample rate. Last error: {last_err!r}")

        # Build cochlear if device_sr != target_sr
        if self._device_sr != self.cfg.sample_rate:
            self._cochlear = CochlearResampler(self._device_sr, self.cfg.sample_rate)
            self._dbg(f"[whisper_audio] cochlear resample: device_sr={self._device_sr} -> target_sr={self.cfg.sample_rate}")
        else:
            self._cochlear = None
            self._dbg(f"[whisper_audio] device_sr matches target_sr={self.cfg.sample_rate}")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._dbg("[whisper_audio] started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

        self._dbg("[whisper_audio] stopped")

    def _run(self) -> None:
        buf = b""
        voiced_frames: List[bytes] = []
        in_speech = False
        voiced_count = 0
        unvoiced_count = 0
        utter_start = 0.0

        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # If device_sr != target_sr, run cochlear first
            if self._cochlear is not None:
                chunk = self._cochlear.push_int16_bytes(chunk)
                if not chunk:
                    continue

            buf += chunk

            # Consume fixed-size frames (ALWAYS in target_sr domain)
            while len(buf) >= self._frame_bytes:
                frame = buf[: self._frame_bytes]
                buf = buf[self._frame_bytes :]

                # RAW MODE: just emit frames; cochlear neuron will resample.
                if self.cfg.raw_only and self.on_audio_raw:
                    try:
                        self.on_audio_raw(frame, self._capture_rate)
                    except Exception as e:
                        self._dbg(f"[whisper_audio] on_audio_raw error: {e}")
                    continue

                # Awake mode: VAD + Whisper (expects cfg.sample_rate framing!)
                is_speech = self._vad.is_speech(frame, self.cfg.sample_rate)

                if not in_speech:
                    if is_speech:
                        voiced_count += 1
                        if voiced_count >= self.cfg.start_trigger_frames:
                            in_speech = True
                            utter_start = time.time()
                            voiced_frames.append(frame)
                            unvoiced_count = 0
                            self._dbg("[whisper_audio] speech start")
                    else:
                        voiced_count = 0
                else:
                    voiced_frames.append(frame)

                    if is_speech:
                        unvoiced_count = 0
                    else:
                        unvoiced_count += 1

                    # End conditions: silence or max duration
                    if (
                        unvoiced_count >= self.cfg.end_silence_frames
                        or (time.time() - utter_start) >= self.cfg.max_utterance_seconds
                    ):
                        self._dbg("[whisper_audio] speech end -> transcribe")
                        self._transcribe_and_emit(voiced_frames)
                        # reset
                        voiced_frames = []
                        in_speech = False
                        voiced_count = 0
                        unvoiced_count = 0

    def _transcribe_and_emit(self, frames: List[bytes]) -> None:
        if not frames:
            return

        pcm = b"".join(frames)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # faster-whisper returns segments; join them
        segments, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
        )

        parts: List[str] = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                parts.append(t)

        text = " ".join(parts).strip()
        if text:
            self._dbg(f"[whisper_audio] heard: {text}")
            if self.on_utterance:
                try:
                    self.on_utterance(text, pcm, self.cfg.sample_rate)
                except Exception as e:
                    self._dbg(f"[whisper_audio] on_utterance error: {e}")
            self.on_transcript(text)
