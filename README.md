# Voice Agent

A local Python voice assistant with wake-word activation, speech-to-text, local agent processing, and offline text-to-speech.

The assistant waits for "Alexa", records the user's request, transcribes it with Moonshine Voice, processes it through the local agent/LLM stack, speaks the response, and stays active for follow-up questions until there is 10 seconds of silence.

---

# Features

- Wake word activation with openWakeWord
- Active conversation sessions after wake
- Optional barge-in interruption while the assistant is speaking
- 10-second inactivity sleep timeout
- Siri-style recording timing with longer start and pause tolerance
- Silero VAD-based speech detection
- On-device speech-to-text using Moonshine Voice
- Local LLM integration (Ollama or llama-cpp-python)
- Offline text-to-speech with pyttsx3 or Piper
- Spoken math calculations
- Voice commands for sleep, repeat, reset, and speech speed
- Short-term conversation memory
- Persistent local memory for user profile, preferences, and facts
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
- Moonshine Voice
- llama-cpp-python (recommended) or Ollama
- pyttsx3 or Piper TTS
- NumPy / SciPy
- FFmpeg

---

# Project Structure

```text
voice_agent/
|-- agent/
|   |-- processor.py
|   |-- math_handler.py
|   |-- memory_store.py
|   `-- llm_handler.py
|-- audio/
|   `-- recorder.py
|-- stt/
|   `-- moonshine_stt.py
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
-> Listen for "Alexa"
-> Active session
-> Record speech until silence
-> Transcribe with Moonshine Voice
-> Process with local agent / LLM
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

## 4. Install and Start LLM Backend

**Option A — Ollama (easier, more RAM):**

```bash
ollama run mistral
```

**Option B — llama-cpp-python (recommended for low-RAM devices):**

```bash
pip install llama-cpp-python
# Download a GGUF model from HuggingFace, e.g. Qwen2.5-1.5B-Instruct Q4_K_M
```

## 5. Run Voice Agent

```bash
python main.py
```

---

# Current Voice Behavior

- Wake phrase: `Alexa`
- Wake detector model: built-in `alexa`
- Wake threshold: `0.5`
- Recorder VAD threshold: `0.35`
- Start-speaking timeout: `4.0` seconds
- End-of-speech silence: `2.5` seconds
- Active-session sleep timeout: `10` seconds
- Max single recording: `45` seconds
- Barge-in threshold: `1200` RMS
- Barge-in grace period: `0.4` seconds

Barge-in is disabled by default to prevent the assistant's own speaker output
from interrupting its response. The default flow is: finish speaking, wait for
the audio buffers to settle, and then listen for the next request. Enable
barge-in only when using headphones or an echo-cancelled audio setup:

```text
VOICE_AGENT_BARGE_IN=true
```

These values are tuned in `wakeword/detector.py`, `audio/recorder.py`, and `main.py`.

Useful speed/config environment variables:

```text
VOICE_AGENT_WAKE_WORD=alexa
VOICE_AGENT_INPUT_DEVICE=Microphone
VOICE_AGENT_OUTPUT_DEVICE=Speakers
VOICE_AGENT_BARGE_IN=false
VOICE_AGENT_STT_LANGUAGE=en
VOICE_AGENT_MOONSHINE_MODEL_ARCH=1
VOICE_AGENT_MOONSHINE_MODEL_PATH=C:\path\to\moonshine-model
OLLAMA_NUM_PREDICT=80
OLLAMA_KEEP_ALIVE=10m
```

`VOICE_AGENT_INPUT_DEVICE` and `VOICE_AGENT_OUTPUT_DEVICE` can be either
a sounddevice index, such as `1`, or a case-insensitive device name fragment,
such as `Yeti` or `Realtek`.

---

# Voice Commands

```text
go to sleep        -> return to wake-word mode
stop               -> return to wake-word mode
stop listening     -> return to wake-word mode
reset conversation -> clear short-term chat and math memory
repeat that        -> repeat the last assistant response
speak slower       -> lower TTS speaking rate
speak faster       -> raise TTS speaking rate
my name is Sam     -> save your name locally
my favorite color is green -> save a preference locally
remember that I am learning Python -> save a conversation fact locally
what do you remember -> summarize saved local memory
shutdown           -> exit the app
exit               -> exit the app
```

---

# Local Memory

Persistent memory is stored as JSON at:

```text
data/memory.json
```

Override the path with:

```text
VOICE_AGENT_MEMORY_PATH=C:\path\to\memory.json
```

The memory file contains separate sections for `profile`, `preferences`, and
conversation `facts`. It is loaded when the agent starts, saved after memory
updates, and included as compact context for LLM responses.

---

# Raspberry Pi / Low-RAM Deployment

The default stack is designed for local execution. For devices with 4–8 GB RAM, use small Moonshine and GGUF models.

## Lightweight Stack

| Component | Default | Lightweight Alternative |
|-----------|---------|------------------------|
| STT | Moonshine Voice | Use the Tiny or Base on-device model |
| VAD | Silero via PyTorch | **Silero ONNX** (onnxruntime only) |
| LLM | Ollama + Mistral 7B | **llama-cpp-python** + small GGUF model |
| TTS | pyttsx3 | **piper-tts** (better voice, still offline) |
| Audio | PyAudio + sounddevice | sounddevice only (drop PyAudio) |

Estimated RAM with the lightweight stack: **~2–2.5 GB**, leaving comfortable headroom on a 4 GB Pi.

## Recommended LLM Models by RAM

### 4 GB RAM (Raspberry Pi 4 / 5 — 4 GB)

| Model | RAM Usage | Notes |
|-------|-----------|-------|
| Qwen2.5-1.5B-Instruct Q4_K_M | ~1.0 GB | Best balance for 4 GB |
| Gemma-2B Q4_K_M | ~1.5 GB | Good general reasoning |
| TinyLlama-1.1B Q4_K_M | ~0.7 GB | Fastest, most basic |

### 8 GB RAM (Raspberry Pi 5 — 8 GB)

| Model | RAM Usage | Notes |
|-------|-----------|-------|
| Phi-3-mini-4k-instruct Q4_K_M | ~2.5 GB | Recommended — strong reasoning |
| Llama-3.2-3B-Instruct Q4_K_M | ~2.0 GB | Great instruction following |
| Gemma-2-2B Q4_K_M | ~1.5 GB | Fast + solid quality |

Download GGUF models from [HuggingFace](https://huggingface.co/models?search=gguf).

## Moonshine STT Setup

Moonshine downloads and caches the appropriate model on first use. To prepare
the English model before running offline:

```bash
moonshine-voice download --stt --language en
```

## Lightweight LLM Setup

```bash
pip install llama-cpp-python

# In llm_handler.py
from llama_cpp import Llama
llm = Llama(model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf", n_threads=4, n_ctx=2048)
```

## Raspberry Pi Performance Tips

- Prefer a Tiny or Base Moonshine model on constrained hardware
- Set `n_threads=4` in llama-cpp-python to use all Pi cores
- Store model files on a USB 3 SSD rather than the SD card — load times are dramatically faster
- Use a USB microphone instead of the 3.5mm jack for cleaner VAD signal
- If speech is cut off too early, lower `vad_threshold` in `audio/recorder.py`

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
User: Alexa
Assistant: Listening...

User: What is Ethereum?
Assistant: Ethereum is a decentralized blockchain platform...

User: Who created it?
Assistant: Ethereum was created by Vitalik Buterin.
```

---

# Notes

- Moonshine Voice downloads model assets on first use unless they are pre-downloaded.
- When using llama-cpp-python, download the GGUF model manually and set the path in config.
- Wake word detection uses openWakeWord with ONNX Runtime.
- Speech start/end detection uses Silero VAD through openWakeWord.
- The assistant stays active after a response and sleeps only after inactivity.
- If speech is cut off too early, lower `vad_threshold` in `audio/recorder.py`.

---

# Future Plans

- Memory editing commands
- Structured tool calling
- Browser automation
- Crypto wallet and hardware wallet integrations
- Raspberry Pi image / one-command setup script
- Config file for hardware profile selection (desktop / Pi 4 / Pi 5)

---

# License

MIT
