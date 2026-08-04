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
from stt.moonshine_stt import MoonshineSTT
from tts.tts_engine import TTSEngine
from agent.processor import AgentProcessor
from wakeword.detector import WakeWordDetector
from utils.logger import (
    elapsed_seconds,
    format_duration,
    log,
    log_timing,
    monotonic_seconds,
)


# Time before assistant goes back to sleep after the last user speech.
SESSION_TIMEOUT = 10

# Give speakers and microphone buffers a moment to settle after TTS.
RESPONSE_COOLDOWN = 1.0

# Desktop defaults stay unchanged. These env vars are optional knobs for
# later low-RAM profiles without changing normal local development.
STT_LANGUAGE = os.environ.get("VOICE_AGENT_STT_LANGUAGE", "en")
MOONSHINE_MODEL_PATH = os.environ.get("VOICE_AGENT_MOONSHINE_MODEL_PATH")
_moonshine_model_arch = os.environ.get("VOICE_AGENT_MOONSHINE_MODEL_ARCH")
MOONSHINE_MODEL_ARCH = int(_moonshine_model_arch) if _moonshine_model_arch else None
WAKE_WORD = os.environ.get("VOICE_AGENT_WAKE_WORD", "alexa")
WAKE_MODEL_PATH = os.environ.get("VOICE_AGENT_WAKE_MODEL")


def _response_flow_details(
    turn_id: int,
    result: str,
    recording_duration: float,
    transcription_duration: float | None = None,
    agent_duration: float | None = None,
    tts_duration: float | None = None,
    cooldown_duration: float | None = None,
    speech_completed: bool | None = None,
) -> str:
    """
    Build consistent summary details for end-to-end response timing logs.
    """

    parts = [
        f"turn={turn_id}",
        f"result={result}",
        f"recording={format_duration(recording_duration)}",
    ]

    if transcription_duration is not None:
        parts.append(
            f"transcription={format_duration(transcription_duration)}"
        )

    if agent_duration is not None:
        parts.append(f"agent={format_duration(agent_duration)}")

    if tts_duration is not None:
        parts.append(f"tts={format_duration(tts_duration)}")

    if cooldown_duration is not None:
        parts.append(f"cooldown={format_duration(cooldown_duration)}")

    if speech_completed is not None:
        parts.append(f"speech_completed={speech_completed}")

    return ", ".join(parts)


def main():

    log("Voice Agent starting")

    log_audio_devices()

    input_device = resolve_input_device()
    output_device = resolve_output_device()

    # Core components
    recorder = AudioRecorder(
        input_device=input_device,
    )

    stt = MoonshineSTT(
        language=STT_LANGUAGE,
        model_arch=MOONSHINE_MODEL_ARCH,
        model_path=MOONSHINE_MODEL_PATH,
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

    last_activity_time = 0.0

    last_response = ""

    turn_id = 0

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

                wake_started_at = monotonic_seconds()

                wakeword.listen()

                log_timing(
                    "Wake flow",
                    wake_started_at,
                    details="result=assistant_active",
                )

                assistant_active = True

                last_activity_time = monotonic_seconds()

                log("Assistant active")

            # ---------------------------------------------
            # Active session
            # ---------------------------------------------

            log("Listening for user speech")

            recording_started_at = monotonic_seconds()

            remaining_session_time = max(
                0.1,
                SESSION_TIMEOUT - elapsed_seconds(last_activity_time)
            )

            audio_path = recorder.record_until_silence(
                no_speech_timeout=min(
                    recorder.no_speech_timeout,
                    remaining_session_time
                )
            )

            recording_duration = elapsed_seconds(recording_started_at)

            # No speech detected
            if audio_path is None:

                log_timing(
                    "Listening window",
                    recording_started_at,
                    details=(
                        "speech_detected=False, "
                        f"remaining_session={format_duration(remaining_session_time)}"
                    ),
                )

                # Session timeout
                if (
                    elapsed_seconds(last_activity_time)
                    > SESSION_TIMEOUT
                ):

                    assistant_active = False

                    wakeword.reset()

                    silent_seconds = elapsed_seconds(last_activity_time)

                    log(
                        (
                            "Assistant sleeping "
                            f"after {format_duration(silent_seconds)} of silence"
                        )
                    )

                continue

            # User spoke
            turn_id += 1

            flow_started_at = recording_started_at

            last_activity_time = monotonic_seconds()

            log(f"Response flow started (turn={turn_id})")

            log_timing(
                "Recording stage",
                recording_started_at,
                details=(
                    f"turn={turn_id}, audio_path={audio_path}, "
                    f"speech_detected=True"
                ),
            )

            # ---------------------------------------------
            # Speech-to-text
            # ---------------------------------------------

            log("Transcribing")

            transcription_started_at = monotonic_seconds()

            text = stt.transcribe(audio_path)

            transcription_duration = elapsed_seconds(transcription_started_at)

            log_timing(
                "Transcription stage",
                transcription_started_at,
                details=f"turn={turn_id}, chars={len(text.strip())}",
            )

            if not text or len(text.strip()) < 2:

                log_timing(
                    "Response flow total",
                    flow_started_at,
                    details=_response_flow_details(
                        turn_id=turn_id,
                        result="empty_transcription",
                        recording_duration=recording_duration,
                        transcription_duration=transcription_duration,
                    ),
                )

                continue

            log(f"You: {text}")

            # ---------------------------------------------
            # Agent response
            # ---------------------------------------------

            agent_started_at = monotonic_seconds()

            response = agent.process(text)

            agent_duration = elapsed_seconds(agent_started_at)

            log_timing(
                "Agent stage",
                agent_started_at,
                details=(
                    f"turn={turn_id}, response_chars={len(response)}, "
                    f"control_response={response.startswith('__')}"
                ),
            )

            if response == "__SLEEP__":

                tts_started_at = monotonic_seconds()

                tts.speak("Going to sleep.")

                tts_duration = elapsed_seconds(tts_started_at)

                log_timing(
                    "TTS stage",
                    tts_started_at,
                    details=(
                        f"turn={turn_id}, response_chars={len('Going to sleep.')}, "
                        "speech_completed=True"
                    ),
                )

                assistant_active = False

                wakeword.reset()

                log("Assistant sleeping by voice command")

                log_timing(
                    "Response flow total",
                    flow_started_at,
                    details=_response_flow_details(
                        turn_id=turn_id,
                        result="sleep_command",
                        recording_duration=recording_duration,
                        transcription_duration=transcription_duration,
                        agent_duration=agent_duration,
                        tts_duration=tts_duration,
                        speech_completed=True,
                    ),
                )

                continue

            if response == "__EXIT__":

                log("Shutdown requested by voice command")

                tts_started_at = monotonic_seconds()

                tts.speak("Goodbye!")

                tts_duration = elapsed_seconds(tts_started_at)

                log_timing(
                    "TTS stage",
                    tts_started_at,
                    details=(
                        f"turn={turn_id}, response_chars={len('Goodbye!')}, "
                        "speech_completed=True"
                    ),
                )

                log_timing(
                    "Response flow total",
                    flow_started_at,
                    details=_response_flow_details(
                        turn_id=turn_id,
                        result="exit_command",
                        recording_duration=recording_duration,
                        transcription_duration=transcription_duration,
                        agent_duration=agent_duration,
                        tts_duration=tts_duration,
                        speech_completed=True,
                    ),
                )

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

            tts_started_at = monotonic_seconds()

            speech_completed = tts.speak(
                response,
                interruptible=True
            )

            tts_duration = elapsed_seconds(tts_started_at)

            log_timing(
                "TTS stage",
                tts_started_at,
                details=(
                    f"turn={turn_id}, response_chars={len(response)}, "
                    f"speech_completed={speech_completed}"
                ),
            )

            if not speech_completed:

                last_activity_time = monotonic_seconds()

                log("Assistant interrupted; listening for barge-in speech")

                log_timing(
                    "Response flow total",
                    flow_started_at,
                    details=_response_flow_details(
                        turn_id=turn_id,
                        result="interrupted",
                        recording_duration=recording_duration,
                        transcription_duration=transcription_duration,
                        agent_duration=agent_duration,
                        tts_duration=tts_duration,
                        speech_completed=False,
                    ),
                )

                continue

            # Small cooldown after speaking before listening for follow-up speech.
            log("Response cooldown")

            cooldown_started_at = monotonic_seconds()

            time.sleep(RESPONSE_COOLDOWN)

            cooldown_duration = elapsed_seconds(cooldown_started_at)

            # Start the silence timeout after the assistant is ready again,
            # not while it is transcribing, thinking, or speaking.
            last_activity_time = monotonic_seconds()

            log_timing(
                "Response flow total",
                flow_started_at,
                details=_response_flow_details(
                    turn_id=turn_id,
                    result="completed",
                    recording_duration=recording_duration,
                    transcription_duration=transcription_duration,
                    agent_duration=agent_duration,
                    tts_duration=tts_duration,
                    cooldown_duration=cooldown_duration,
                    speech_completed=True,
                ),
            )

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
