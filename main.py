"""
Main entry point for the voice agent.

Flow:
Mic -> Speech-to-Text -> Agent -> Text-to-Speech
"""

import time

from audio.recorder import AudioRecorder
from stt.whisper_stt import WhisperSTT
from tts.tts_engine import TTSEngine
from agent.processor import AgentProcessor
from wakeword.detector import WakeWordDetector
from utils.logger import log

wakeword = WakeWordDetector()

def main():

    print("\n🎙️ Voice Agent Starting...")
    print("=" * 45)

    # Core components
    recorder = AudioRecorder()

    # Whisper model options:
    # tiny | base | small | medium
    stt = WhisperSTT(model_size="small")

    tts = TTSEngine()

    agent = AgentProcessor()

    print("✅ System ready. Say something...\n")

    # Startup voice check
    tts.speak("Voice agent ready. How can I help you?")

    while True:

        try:
            wakeword.listen()
            # Wait for user input
            print("🎤 Listening...")

            audio_path = recorder.record_until_silence()

            if audio_path is None:
                continue

            # Convert speech to text
            print("📝 Transcribing...")

            text = stt.transcribe(audio_path)

            if not text or len(text.strip()) < 2:
                print("(no speech detected)\n")
                continue

            print(f"👤 You: {text}")

            # Generate response
            response = agent.process(text)

            print(f"🤖 Agent: {response}\n")

            # Speak response
            tts.speak(response)

            # Small cooldown before next cycle
            time.sleep(0.3)

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