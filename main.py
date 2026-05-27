"""
Main entry point for the voice agent.

Flow:
Wake Word
-> Speech-to-Text
-> LLM / Agent
-> Text-to-Speech
"""

import time

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


def main():

    log("Voice Agent starting")

    # Core components
    recorder = AudioRecorder()

    stt = WhisperSTT(
        model_size="small"
    )

    tts = TTSEngine()

    agent = AgentProcessor()

    wakeword = WakeWordDetector()

    log("System ready")

    tts.speak(
        "Voice agent ready."
    )

    # Assistant state
    assistant_active = False

    last_activity_time = 0

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

            audio_path = recorder.record_until_silence()

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

            log(f"Agent: {response}")

            # ---------------------------------------------
            # Text-to-speech
            # ---------------------------------------------

            tts.speak(response)

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
