"""
Local LLM handler using Ollama.
"""

from ollama import chat


SYSTEM_PROMPT = """
You are a helpful voice assistant.

Keep responses:
- short
- conversational
- natural for speech
"""


def ask_llm(prompt: str) -> str:

    try:

        response = chat(
            model="mistral",

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        return response["message"]["content"]

    except Exception as e:

        return f"LLM error: {e}"