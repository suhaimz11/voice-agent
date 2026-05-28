"""
Offline text-to-speech engine.

Uses pyttsx3 for local speech synthesis
without requiring any external API.
"""

import math
import threading
import time

import numpy as np
import pyaudio
import pyttsx3

from utils.logger import log


class TTSEngine:

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
        voice_index: int = 0,
        barge_in_threshold: float = 1200,
        barge_in_grace_seconds: float = 0.4,
        barge_in_duration: float = 0.15,
    ):

        # Speech settings
        self.rate = rate

        self.volume = volume

        self.voice_index = voice_index

        # Interruption settings. Raise the threshold if speaker audio
        # triggers false interruptions through the microphone.
        self.barge_in_threshold = barge_in_threshold

        self.barge_in_grace_seconds = barge_in_grace_seconds

        self.barge_in_duration = barge_in_duration

        self.barge_in_sample_rate = 16000

        self.barge_in_chunk_size = 1024

        log("TTSEngine initialized")

    # ---------------------------------------------------------
    def speak(
        self,
        text: str,
        interruptible: bool = False
    ) -> bool:
        """
        Convert text into speech output.

        Returns True when speech finishes normally and False when
        interrupted by barge-in speech.
        """

        if not text:
            return True

        try:

            if interruptible:
                return self._speak_interruptible(text)

            return self._speak_blocking(text)

        except Exception as e:

            log(
                f"TTS error: {e}",
                level="warning",
                exc_info=True
            )

            # Fallback if audio output fails
            log(f"TTS fallback text: {text}")

            return True

    # ---------------------------------------------------------
    def _configure_engine(self):

        # Create a fresh engine instance
        # to avoid Windows audio lock issues
        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            self.rate
        )

        engine.setProperty(
            "volume",
            self.volume
        )

        voices = engine.getProperty("voices")

        # Select configured system voice
        if voices and self.voice_index < len(voices):

            engine.setProperty(
                "voice",
                voices[self.voice_index].id
            )

        return engine

    # ---------------------------------------------------------
    def _speak_blocking(self, text: str) -> bool:

        started_at = time.time()

        engine = self._configure_engine()

        log(f"TTS started: {text}")

        engine.say(text)

        engine.runAndWait()

        engine.stop()

        log(
            f"TTS finished in {time.time() - started_at:.1f}s"
        )

        return True

    # ---------------------------------------------------------
    def _speak_interruptible(self, text: str) -> bool:

        started_at = time.time()

        pa = pyaudio.PyAudio()

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.barge_in_sample_rate,
            input=True,
            frames_per_buffer=self.barge_in_chunk_size,
        )

        chunk_seconds = (
            self.barge_in_chunk_size / self.barge_in_sample_rate
        )

        chunks_needed = max(
            1,
            math.ceil(self.barge_in_duration / chunk_seconds)
        )

        voice_chunks = 0

        interrupted = False

        max_rms = 0.0

        speech_done = threading.Event()

        speech_ready = threading.Event()

        speech_error = []

        engine_holder = {
            "engine": None
        }

        log(f"TTS started: {text}")

        def run_speech():

            try:
                engine = self._configure_engine()

                engine_holder["engine"] = engine

                speech_ready.set()

                engine.say(text)

                engine.runAndWait()

            except Exception as e:
                speech_error.append(e)

            finally:
                speech_ready.set()

                speech_done.set()

        speech_thread = threading.Thread(
            target=run_speech,
            daemon=True
        )

        speech_thread.start()

        speech_ready.wait(timeout=2.0)

        try:

            while not speech_done.is_set():

                if (
                    time.time() - started_at
                    >= self.barge_in_grace_seconds
                ):

                    data = stream.read(
                        self.barge_in_chunk_size,
                        exception_on_overflow=False,
                    )

                    arr = np.frombuffer(
                        data,
                        dtype=np.int16
                    ).astype(np.float32)

                    rms = float(
                        np.sqrt(np.mean(arr ** 2))
                    )

                    max_rms = max(
                        max_rms,
                        rms
                    )

                    if rms >= self.barge_in_threshold:
                        voice_chunks += 1
                    else:
                        voice_chunks = max(
                            0,
                            voice_chunks - 1
                        )

                    if voice_chunks >= chunks_needed:

                        interrupted = True

                        engine = engine_holder["engine"]

                        if engine is not None:
                            engine.stop()

                        speech_done.wait(timeout=1.0)

                        log(
                            (
                                "TTS interrupted by barge-in "
                                f"(rms={rms:.0f}, "
                                f"threshold={self.barge_in_threshold:.0f})"
                            )
                        )

                        break

                else:
                    time.sleep(0.02)

        finally:

            engine = engine_holder["engine"]

            if engine is not None:
                engine.stop()

            speech_thread.join(timeout=1.0)

            stream.stop_stream()

            stream.close()

            pa.terminate()

        if speech_error:
            raise speech_error[0]

        if interrupted:
            return False

        log(
            (
                f"TTS finished in {time.time() - started_at:.1f}s "
                f"(max_barge_in_rms={max_rms:.0f}, "
                f"threshold={self.barge_in_threshold:.0f})"
            )
        )

        return True

    # ---------------------------------------------------------
    def list_voices(self):
        """
        Print available system voices.
        """

        engine = pyttsx3.init()

        voices = engine.getProperty("voices")

        for index, voice in enumerate(voices):

            log(f"[{index}] {voice.name} - {voice.id}")

        engine.stop()
