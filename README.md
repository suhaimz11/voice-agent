# Voice Agent

A local AI voice assistant built with Python.

The assistant listens through the microphone, converts speech into text using Whisper, processes requests using a local LLM through Ollama, and replies back using text-to-speech.

---

# Features

- Wake word activation
- Active conversation sessions
- Real-time microphone input
- Speech-to-text using Whisper
- Local LLM integration with Ollama
- Context-aware conversations
- Short-term conversation memory
- Offline text-to-speech
- Spoken math calculations
- Silence detection
- Fully local execution
- Modular architecture

---

# Tech Stack

- Python
- PyAudio
- Whisper
- Ollama
- Mistral
- openWakeWord
- pyttsx3
- NumPy
- FFmpeg

---

# Project Structure

```bash
voice_agent/
│
├── agent/
│   ├── processor.py
│   ├── math_handler.py
│   └── llm_handler.py
│
├── audio/
│   └── recorder.py
│
├── stt/
│   └── whisper_stt.py
│
├── tts/
│   └── tts_engine.py
│
├── wakeword/
│   └── detector.py
│
├── utils/
│   └── logger.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# How It Works

```text
Wake Word
→ Speech-to-Text
→ Local LLM
→ Text-to-Speech
→ Speaker
```

---

# Setup

## 1. Clone Repository

```bash
git clone https://github.com/Suhaimz11/voice-agent.git

cd voice-agent
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

---

## 3. Activate Virtual Environment

```bash
source .venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Install FFmpeg

Verify FFmpeg installation:

```bash
ffmpeg -version
```

---

## 6. Install Ollama

Pull the Mistral model locally:

```bash
ollama run mistral
```

---

## 7. Run Voice Agent

```bash
python main.py
```

---

# Example

```text
User: Hey Jarvis

Assistant: Listening...

User: What is Ethereum?

Assistant: Ethereum is a decentralized blockchain platform...
```

```text
User: Who created it?

Assistant: Ethereum was created by Vitalik Buterin.
```

```text
User: How old is he?

Assistant: Vitalik Buterin is 31 years old.
```

---

# Notes

- Whisper downloads model weights on first launch
- Ollama must be installed locally
- Runs completely offline
- Recommended Python version: 3.11
- Wake word detection uses openWakeWord
- Active session automatically closes after inactivity

---

# Future Plans

- Structured tool calling
- Crypto wallet integration
- Hardware wallet support
- Voice confirmations for transactions
- Browser automation
- Local memory persistence

---

# License

MIT