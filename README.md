# Voice Agent

A local Python voice assistant with wake-word activation, speech-to-text, local agent processing, and offline text-to-speech.

The assistant waits for "Alexa", records the user's request, transcribes it with Whisper, processes it through the local agent/LLM stack, speaks the response, and stays active for follow-up questions until there is 10 seconds of silence.

---

# Features

- Wake word activation with openWakeWord
- Active conversation sessions after wake
- Barge-in interruption while the assistant is speaking
- 10-second inactivity sleep timeout
- Siri-style recording timing with longer start and pause tolerance
- Silero VAD-based speech detection
- Speech-to-text using Whisper
- Local LLM integration (Ollama or llama-cpp-python)
- Offline text-to-speech with pyttsx3 or Piper
- Spoken math calculations
- Voice commands for sleep, repeat, reset, and speech speed
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
- faster-whisper (recommended) or Whisper
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
-> Listen for "Alexa"
-> Active session
-> Record speech until silence
-> Transcribe with Whisper
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

## 4. Install FFmpeg

Whisper requires FFmpeg. Verify it is available:

```bash
ffmpeg -version
```

## 5. Install and Start LLM Backend

**Option A — Ollama (easier, more RAM):**

```bash
ollama run mistral
```

**Option B — llama-cpp-python (recommended for low-RAM devices):**

```bash
pip install llama-cpp-python
# Download a GGUF model from HuggingFace, e.g. Qwen2.5-1.5B-Instruct Q4_K_M
```

## 6. Run Voice Agent

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

These values are tuned in `wakeword/detector.py`, `audio/recorder.py`, and `main.py`.

Useful speed/config environment variables:

```text
VOICE_AGENT_WAKE_WORD=alexa
VOICE_AGENT_STT_MODEL=small
VOICE_AGENT_STT_BEAM_SIZE=1
VOICE_AGENT_STT_BEST_OF=1
VOICE_AGENT_STT_VAD_FILTER=false
OLLAMA_NUM_PREDICT=80
OLLAMA_KEEP_ALIVE=10m
```

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
shutdown           -> exit the app
exit               -> exit the app
```

---

# Raspberry Pi / Low-RAM Deployment

The default stack (PyTorch + Whisper + Ollama) is too heavy for devices with 4–8 GB RAM. Use the lightweight alternatives below.

## Lightweight Stack

| Component | Default | Lightweight Alternative |
|-----------|---------|------------------------|
| STT | openai-whisper + PyTorch | **faster-whisper** (no PyTorch needed) |
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

## Lightweight STT Setup

```bash
pip install faster-whisper

# In whisper_stt.py — use tiny.en for English-only (fastest, ~150 MB)
from faster_whisper import WhisperModel
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
```

## Lightweight LLM Setup

```bash
pip install llama-cpp-python

# In llm_handler.py
from llama_cpp import Llama
llm = Llama(model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf", n_threads=4, n_ctx=2048)
```

## Raspberry Pi Performance Tips

- Use `compute_type="int8"` in faster-whisper for ARM NEON speedup
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

- faster-whisper downloads model weights on first launch.
- When using llama-cpp-python, download the GGUF model manually and set the path in config.
- Wake word detection uses openWakeWord with ONNX Runtime.
- Speech start/end detection uses Silero VAD through openWakeWord.
- The assistant stays active after a response and sleeps only after inactivity.
- If speech is cut off too early, lower `vad_threshold` in `audio/recorder.py`.

---

# Future Plans

- Persistent local memory
- Structured tool calling
- Browser automation
- Crypto wallet and hardware wallet integrations
- Raspberry Pi image / one-command setup script
- Config file for hardware profile selection (desktop / Pi 4 / Pi 5)

---

# License

MIT
