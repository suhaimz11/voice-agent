"""
Voice Agent - Main Entry Point
Pipeline: Mic → PyAudio → Whisper → Agent → TTS → Speaker
"""

import time
from audio.recorder import AudioRecorder
from stt.whisper_stt import WhisperSTT
from tts.tts_engine import TTSEngine
from agent.processor import AgentProcessor
from utils.logger import log


def main():
    print("\n🎙️  Voice Agent Starting...")
    print("=" * 45)

    # --- Initialize components ---
    recorder  = AudioRecorder()
    stt       = WhisperSTT(model_size="base")   # tiny | base | small | medium
    tts       = TTSEngine()
    agent     = AgentProcessor()

    print("✅  All systems ready. Say something! (Ctrl+C to quit)\n")
    tts.speak("Voice agent ready. How can I help you?")

    # --- Main loop ---
    while True:
        try:
            # 1. Record until silence
            print("🎤  Listening...")
            audio_path = recorder.record_until_silence()

            if audio_path is None:
                continue

            # 2. Transcribe with Whisper
            print("📝  Transcribing...")
            text = stt.transcribe(audio_path)

            if not text or len(text.strip()) < 2:
                print("    (no speech detected)\n")
                continue

            print(f"👤  You: {text}")

            # 3. Process through agent
            response = agent.process(text)
            print(f"🤖  Agent: {response}\n")

            # 4. Speak the response
            tts.speak(response)

            # Small pause before listening again
            time.sleep(0.3)

        except KeyboardInterrupt:
            print("\n\n👋  Shutting down...")
            tts.speak("Goodbye!")
            break
        except Exception as e:
            log(f"Main loop error: {e}", level="error")


if __name__ == "__main__":
    main()
