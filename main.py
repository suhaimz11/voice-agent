"""
Main entry point for the voice agent.

Flow:
Wake Word
-> Speech-to-Text
-> LLM / Agent
-> Text-to-Speech
"""

import os
import time

from audio.devices import (
    log_audio_devices,
    resolve_input_device,
    resolve_output_device,
)
from audio.recorder import AudioRecorder
from stt.whisper_stt import WhisperSTT
from tts.tts_engine import TTSEngine
from agent.processor import AgentProcessor
from wakeword.detector import WakeWordDetector
from utils.logger import log


# Time before assistant goes back to sleep after the last user speech.
SESSION_TIMEOUT = 10

# Give speakers and microphone buffers a moment to settle after TTS.
RESPONSE_COOLDOWN = 1.0

# Desktop defaults stay unchanged. These env vars are optional knobs for
# later low-RAM profiles without changing normal local development.
STT_MODEL_SIZE = os.environ.get("VOICE_AGENT_STT_MODEL", "small")
STT_DEVICE = os.environ.get("VOICE_AGENT_STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.environ.get("VOICE_AGENT_STT_COMPUTE_TYPE", "int8")
WAKE_WORD = os.environ.get("VOICE_AGENT_WAKE_WORD", "alexa")
WAKE_MODEL_PATH = os.environ.get("VOICE_AGENT_WAKE_MODEL")


def main():

    log("Voice Agent starting")

    log_audio_devices()

    input_device = resolve_input_device()
    output_device = resolve_output_device()

    # Core components
    recorder = AudioRecorder(
        input_device=input_device,
    )

    stt = WhisperSTT(
        model_size=STT_MODEL_SIZE,
        device=STT_DEVICE,
        compute_type=STT_COMPUTE_TYPE,
    )

    tts = TTSEngine(
        input_device=input_device,
        output_device=output_device,
    )

    agent = AgentProcessor()

    wakeword = WakeWordDetector(
        wake_word=WAKE_WORD,
        model_path=WAKE_MODEL_PATH,
        input_device=input_device,
    )

    log("System ready")

    tts.speak(
        "Voice agent ready."
    )

    # Assistant state
    assistant_active = False

    last_activity_time = 0

    last_response = ""

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------

    while True:

        try:

            # ---------------------------------------------
            # Sleep mode
            # ---------------------------------------------

            if not assistant_active:

                log("Waiting for wake word")

                wakeword.listen()

                assistant_active = True

                last_activity_time = time.time()

                log("Assistant active")

            # ---------------------------------------------
            # Active session
            # ---------------------------------------------

            log("Listening for user speech")

            remaining_session_time = max(
                0.1,
                SESSION_TIMEOUT - (time.time() - last_activity_time)
            )

            audio_path = recorder.record_until_silence(
                no_speech_timeout=min(
                    recorder.no_speech_timeout,
                    remaining_session_time
                )
            )

            # No speech detected
            if audio_path is None:

                # Session timeout
                if (
                    time.time() - last_activity_time
                    > SESSION_TIMEOUT
                ):

                    assistant_active = False

                    wakeword.reset()

                    silent_seconds = time.time() - last_activity_time

                    log(
                        (
                            "Assistant sleeping "
                            f"after {silent_seconds:.1f}s of silence"
                        )
                    )

                continue

            # User spoke
            last_activity_time = time.time()

            # ---------------------------------------------
            # Speech-to-text
            # ---------------------------------------------

            log("Transcribing")

            text = stt.transcribe(audio_path)

            if not text or len(text.strip()) < 2:

                continue

            log(f"You: {text}")

            # ---------------------------------------------
            # Agent response
            # ---------------------------------------------

            response = agent.process(text)

            if response == "__SLEEP__":

                tts.speak("Going to sleep.")

                assistant_active = False

                wakeword.reset()

                log("Assistant sleeping by voice command")

                continue

            if response == "__EXIT__":

                log("Shutdown requested by voice command")

                tts.speak("Goodbye!")

                break

            if response == "__RESET_CONVERSATION__":

                last_response = ""

                response = "Conversation reset."

            elif response == "__REPEAT__":

                response = (
                    last_response
                    if last_response
                    else "I do not have anything to repeat yet."
                )

            elif response == "__SPEAK_SLOWER__":

                tts.adjust_rate(-20)

                response = "I will speak slower."

            elif response == "__SPEAK_FASTER__":

                tts.adjust_rate(20)

                response = "I will speak faster."

            log(f"Agent: {response}")

            last_response = response

            # ---------------------------------------------
            # Text-to-speech
            # ---------------------------------------------

            speech_completed = tts.speak(
                response,
                interruptible=True
            )

            if not speech_completed:

                last_activity_time = time.time()

                log("Assistant interrupted; listening for barge-in speech")

                continue

            # Small cooldown after speaking before listening for follow-up speech.
            log("Response cooldown")

            time.sleep(RESPONSE_COOLDOWN)

            # Start the silence timeout after the assistant is ready again,
            # not while it is transcribing, thinking, or speaking.
            last_activity_time = time.time()

        except KeyboardInterrupt:

            log("Shutting down")

            tts.speak("Goodbye!")

            break

        except Exception as e:

            log(
                f"Main loop error: {e}",
                level="error",
                exc_info=True
            )


if __name__ == "__main__":

    main()
