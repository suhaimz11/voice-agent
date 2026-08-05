"""
Main entry point for the voice agent.

Flow:
Wake Word
-> Speech-to-Text
-> LLM / Agent
-> Text-to-Speech
"""

import os
import re
import time
from enum import Enum, auto

from audio.devices import (
    log_audio_devices,
    resolve_input_device,
    resolve_output_device,
)
from audio.recorder import AudioRecorder
from stt.moonshine_stt import MoonshineSTT
from tts.tts_engine import TTSEngine
from agent.processor import AgentProcessor
from agent.llm_handler import warm_llm
from config import load_runtime_profile
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
class AssistantState(Enum):
    SLEEPING = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    FOLLOW_UP = auto()
    SHUTTING_DOWN = auto()
    ERROR = auto()


ALLOWED_STATE_TRANSITIONS = {
    AssistantState.SLEEPING: {
        AssistantState.LISTENING,
        AssistantState.SHUTTING_DOWN,
        AssistantState.ERROR,
    },
    AssistantState.LISTENING: {
        AssistantState.PROCESSING,
        AssistantState.SPEAKING,
        AssistantState.SLEEPING,
        AssistantState.SHUTTING_DOWN,
        AssistantState.ERROR,
    },
    AssistantState.PROCESSING: {
        AssistantState.SPEAKING,
        AssistantState.SLEEPING,
        AssistantState.SHUTTING_DOWN,
        AssistantState.ERROR,
    },
    AssistantState.SPEAKING: {
        AssistantState.LISTENING,
        AssistantState.FOLLOW_UP,
        AssistantState.SLEEPING,
        AssistantState.SHUTTING_DOWN,
        AssistantState.ERROR,
    },
    AssistantState.FOLLOW_UP: {
        AssistantState.LISTENING,
        AssistantState.SLEEPING,
        AssistantState.SHUTTING_DOWN,
        AssistantState.ERROR,
    },
    AssistantState.ERROR: {
        AssistantState.LISTENING,
        AssistantState.SLEEPING,
        AssistantState.SHUTTING_DOWN,
    },
    AssistantState.SHUTTING_DOWN: set(),
}


STT_LANGUAGE = os.environ.get("VOICE_AGENT_STT_LANGUAGE", "en")
MOONSHINE_MODEL_PATH = os.environ.get("VOICE_AGENT_MOONSHINE_MODEL_PATH")
_moonshine_model_arch = os.environ.get("VOICE_AGENT_MOONSHINE_MODEL_ARCH")
MOONSHINE_MODEL_ARCH = int(_moonshine_model_arch) if _moonshine_model_arch else None
WAKE_WORD = os.environ.get("VOICE_AGENT_WAKE_WORD", "alexa")
WAKE_MODEL_PATH = os.environ.get("VOICE_AGENT_WAKE_MODEL")

INCOMPLETE_TRANSCRIPT = re.compile(
    r"^(what is|what are|who is|where is|when is|why is|how do|how can|"
    r"can you|could you|would you|tell me|explain)\s*$",
    re.I,
)


def _transcript_needs_retry(text: str) -> bool:
    cleaned = text.strip().rstrip(".,!?;:")
    return len(cleaned) < 2 or bool(INCOMPLETE_TRANSCRIPT.match(cleaned))


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

    profile = load_runtime_profile()
    log(f"Runtime profile: {profile.name}")

    log_audio_devices()

    input_device = resolve_input_device()
    output_device = resolve_output_device()

    # Core components
    recorder = AudioRecorder(
        silence_duration=profile.end_silence_seconds,
        no_speech_timeout=profile.speech_start_timeout,
        max_record_seconds=profile.maximum_utterance_seconds,
        minimum_speech_seconds=profile.minimum_speech_seconds,
        pre_speech_seconds=profile.pre_speech_seconds,
        input_device=input_device,
    )

    stt = MoonshineSTT(
        language=STT_LANGUAGE,
        model_arch=(
            MOONSHINE_MODEL_ARCH
            if MOONSHINE_MODEL_ARCH is not None
            else profile.moonshine_model_arch
        ),
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

    warm_llm()

    log("System ready")

    tts.speak(
        "Voice agent ready."
    )

    # Assistant state
    state = AssistantState.SLEEPING

    def transition(next_state: AssistantState) -> None:
        nonlocal state
        if next_state == state:
            return
        if next_state not in ALLOWED_STATE_TRANSITIONS[state]:
            raise RuntimeError(
                f"Invalid assistant state transition: {state.name} -> {next_state.name}"
            )
        log(f"State: {state.name} -> {next_state.name}", level="debug")
        state = next_state

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

            if state == AssistantState.SLEEPING:

                log("Waiting for wake word")

                wake_started_at = monotonic_seconds()

                wakeword.listen()

                log_timing(
                    "Wake flow",
                    wake_started_at,
                    details="result=assistant_active",
                )

                transition(AssistantState.LISTENING)

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

            transition(AssistantState.LISTENING)
            stt.start_stream()
            capture = recorder.stream_until_silence(
                stt.add_audio,
                no_speech_timeout=min(recorder.no_speech_timeout, remaining_session_time),
            )

            recording_duration = elapsed_seconds(recording_started_at)

            # No speech detected
            if not capture.speech_detected:
                stt.close_stream()

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

                    transition(AssistantState.SLEEPING)

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
                    f"turn={turn_id}, streaming=True, "
                    f"speech_detected=True"
                ),
            )

            # ---------------------------------------------
            # Speech-to-text
            # ---------------------------------------------

            log("Transcribing")

            transcription_started_at = monotonic_seconds()

            text = stt.finish_stream()

            transcription_duration = elapsed_seconds(transcription_started_at)

            log_timing(
                "Transcription stage",
                transcription_started_at,
                details=f"turn={turn_id}, chars={len(text.strip())}",
            )

            if _transcript_needs_retry(text):

                log_timing(
                    "Response flow total",
                    flow_started_at,
                    details=_response_flow_details(
                        turn_id=turn_id,
                        result="transcription_retry",
                        recording_duration=recording_duration,
                        transcription_duration=transcription_duration,
                    ),
                )

                transition(AssistantState.SPEAKING)
                tts.speak("I didn't catch that. Please try again.")
                last_activity_time = monotonic_seconds()
                transition(AssistantState.FOLLOW_UP)
                continue

            log(f"You: {text}")

            # ---------------------------------------------
            # Agent response
            # ---------------------------------------------

            transition(AssistantState.PROCESSING)
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

                transition(AssistantState.SPEAKING)
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

                transition(AssistantState.SLEEPING)

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

                transition(AssistantState.SPEAKING)
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

                transition(AssistantState.SHUTTING_DOWN)
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

            transition(AssistantState.SPEAKING)
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

                transition(AssistantState.LISTENING)

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

            time.sleep(profile.response_cooldown_seconds)

            cooldown_duration = elapsed_seconds(cooldown_started_at)

            # Start the silence timeout after the assistant is ready again,
            # not while it is transcribing, thinking, or speaking.
            last_activity_time = monotonic_seconds()
            transition(AssistantState.FOLLOW_UP)

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

            transition(AssistantState.SHUTTING_DOWN)
            log("Shutting down")

            tts.speak("Goodbye!")

            break

        except Exception as e:

            if state != AssistantState.SHUTTING_DOWN:
                transition(AssistantState.ERROR)
            stt.close_stream()
            log(
                f"Main loop error: {e}",
                level="error",
                exc_info=True
            )


if __name__ == "__main__":

    main()
