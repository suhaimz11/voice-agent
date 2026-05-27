"""
Main entry point for the voice agent.

Flow:
Wake Word
→ Speech-to-Text
→ LLM / Agent
→ Text-to-Speech
"""

import time

from audio.recorder import AudioRecorder
from stt.whisper_stt import WhisperSTT
from tts.tts_engine import TTSEngine
from agent.processor import AgentProcessor
from wakeword.detector import WakeWordDetector
from utils.logger import log


# Time before assistant goes back to sleep
SESSION_TIMEOUT = 6


def main():

    print("\n🎙️ Voice Agent Starting...")
    print("=" * 45)

    # Core components
    recorder = AudioRecorder()

    stt = WhisperSTT(
        model_size="small"
    )

    tts = TTSEngine()

    agent = AgentProcessor()

    wakeword = WakeWordDetector()

    print("✅ System ready.\n")

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

                print("👂 Waiting for wake word...")

                wakeword.listen()

                assistant_active = True

                last_activity_time = time.time()

                print("🟢 Assistant active")

            # ---------------------------------------------
            # Active session
            # ---------------------------------------------

            print("🎤 Listening...")

            audio_path = recorder.record_until_silence()

            # No speech detected
            if audio_path is None:

                # Session timeout
                if (
                    time.time() - last_activity_time
                    > SESSION_TIMEOUT
                ):

                    assistant_active = False

                    print("😴 Assistant sleeping")

                continue

            # User spoke
            last_activity_time = time.time()

            # ---------------------------------------------
            # Speech-to-text
            # ---------------------------------------------

            print("📝 Transcribing...")

            text = stt.transcribe(audio_path)

            if not text or len(text.strip()) < 2:

                continue

            print(f"👤 You: {text}")

            # ---------------------------------------------
            # Agent response
            # ---------------------------------------------

            response = agent.process(text)

            print(f"🤖 Agent: {response}\n")

            # ---------------------------------------------
            # Text-to-speech
            # ---------------------------------------------

            tts.speak(response)

            # Small cooldown after speaking
            print("⏳ Cooldown...")

            time.sleep(2)

        except KeyboardInterrupt:

            print("\n👋 Shutting down...")

            tts.speak("Goodbye!")

            break

        except Exception as e:

            log(
                f"Main loop error: {e}",
                level="error"
            )


if __name__ == "__main__":

    main()