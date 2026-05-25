# Voice Agent

A local AI voice assistant built with Python.

The assistant listens through the microphone, converts speech into text using Whisper, processes requests using a local LLM through Ollama, and replies back using text-to-speech.

---

# Features

- Real-time microphone input
- Speech-to-text using Whisper
- Local LLM integration with Ollama
- Context-aware conversations
- Short-term memory support
- Offline text-to-speech
- Spoken math calculations
- Silence detection
- Modular architecture

---

# Tech Stack

- Python
- PyAudio
- Whisper
- Ollama
- Mistral
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
Mic
→ PyAudio
→ Whisper STT
→ Ollama (Mistral)
→ TTS
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
You: What is 25 plus 17?
Agent: The answer is 42.
```

```text
You: My name is Suhaim.
Agent: Nice to meet you, Suhaim.

You: What is my name?
Agent: Your name is Suhaim.
```

---

# Notes

- Whisper downloads model weights on first launch
- Ollama must be installed locally
- Recommended Python version: 3.11
- Works fully offline

---

# License

MIT