"""
Local LLM handler using Ollama.

Handles:
- conversational responses
- short-term chat memory
- multi-turn interactions
"""

from ollama import chat


SYSTEM_PROMPT = """
You are a local AI voice assistant.

You can remember information shared during the current conversation.

If the user tells you their name,
remember it and use it naturally later.

Keep responses:
- short
- conversational
- natural for speech

Avoid long paragraphs.
"""


# Stores ongoing conversation history
conversation_history = []


def ask_llm(prompt: str) -> str:
    """
    Send user prompt to local Ollama model
    and return the assistant response.
    """

    try:

        # Build message list
        messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            }

        ]

        # Add previous conversation
        messages.extend(
            conversation_history
        )

        # Add latest user message
        messages.append(

            {
                "role": "user",
                "content": prompt,
            }

        )

        # Generate response
        response = chat(

            model="mistral",

            messages=messages

        )

        reply = response["message"]["content"]

        # Save conversation memory
        conversation_history.append(

            {
                "role": "user",
                "content": prompt,
            }

        )

        conversation_history.append(

            {
                "role": "assistant",
                "content": reply,
            }

        )

        # Prevent memory from growing forever
        MAX_HISTORY = 12

        if len(conversation_history) > MAX_HISTORY:

            del conversation_history[
                :len(conversation_history) - MAX_HISTORY
            ]

        return reply.strip()

    except Exception as e:

        return f"LLM error: {e}"