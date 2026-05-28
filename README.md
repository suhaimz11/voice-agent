# Voice Agent

A local Python voice assistant with wake-word activation, speech-to-text, local agent processing, and offline text-to-speech.

The assistant waits for "Hey Jarvis", records the user's request, transcribes it with Whisper, processes it through the local agent/Ollama stack, speaks the response, and stays active for follow-up questions until there is 10 seconds of silence.

---

# Features

- Wake word activation with openWakeWord
- Active conversation sessions after wake
- Barge-in interruption while the assistant is speaking
- 10-second inactivity sleep timeout
- Siri-style recording timing with longer start and pause tolerance
- Speech-to-text using Whisper
- Local LLM integration through Ollama
- Offline text-to-speech with pyttsx3
- Spoken math calculations
- Short-term conversation memory
- File and console logging
- Fully local execution
- Modular architecture

---

# Tech Stack

- Python 3.11 recommended
- PyAudio
- sounddevice
- openWakeWord
- ONNX Runtime
- Whisper
- PyTorch
- Ollama
- Mistral
- pyttsx3
- NumPy / SciPy
- FFmpeg

---

# Project Structure

```text
voice_agent/
|-- agent/
|   |-- processor.py
|   |-- math_handler.py
|   `-- llm_handler.py
|-- audio/
|   `-- recorder.py
|-- stt/
|   `-- whisper_stt.py
|-- tts/
|   `-- tts_engine.py
|-- wakeword/
|   `-- detector.py
|-- utils/
|   `-- logger.py
|-- main.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```

---

# How It Works

```text
Sleep mode
-> Listen for "Hey Jarvis"
-> Active session
-> Record speech until silence
-> Transcribe with Whisper
-> Process with local agent / Ollama
-> Speak response
-> Listen for follow-up speech
-> Sleep after 10 seconds of silence
```

---

# Setup

## 1. Create Virtual Environment

```bash
python -m venv .venv
```

## 2. Activate Virtual Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install FFmpeg

Whisper requires FFmpeg. Verify it is available:

```bash
ffmpeg -version
```

## 5. Install Ollama

Install Ollama, then pull the local model:

```bash
ollama run mistral
```

## 6. Run Voice Agent

```bash
python main.py
```

---

# Current Voice Behavior

- Wake phrase: "Hey Jarvis"
- Wake detector model: `hey_jarvis`
- Wake threshold: `0.5`
- Start-speaking timeout: `4.0` seconds
- End-of-speech silence: `2.5` seconds
- Active-session sleep timeout: `10` seconds
- Max single recording: `45` seconds
- Barge-in threshold: `1200` RMS
- Barge-in grace period: `0.4` seconds

These values are tuned in `wakeword/detector.py`, `audio/recorder.py`, and `main.py`.

---

# Logs

Runtime logs are written to:

```text
logs/voice_agent.log
```

Logs include startup, wake detection, active/sleep transitions, recording timing, transcription output, TTS timing, and errors with tracebacks.

The `logs/` directory is ignored by git.

---

# Example

```text
User: Hey Jarvis
Assistant: Listening...

User: What is Ethereum?
Assistant: Ethereum is a decentralized blockchain platform...

User: Who created it?
Assistant: Ethereum was created by Vitalik Buterin.
```

---

# Notes

- Whisper downloads model weights on first launch.
- Ollama must be running locally for LLM responses.
- Wake word detection uses openWakeWord with ONNX Runtime.
- The assistant stays active after a response and sleeps only after inactivity.
- If speech is cut off too early, lower `silence_threshold` in `audio/recorder.py`.

---

# Future Plans

- Better VAD-based speech detection
- Explicit voice commands such as "go to sleep" and "reset conversation"
- Persistent local memory
- Structured tool calling
- Browser automation
- Crypto wallet and hardware wallet integrations

---

# License

MIT
