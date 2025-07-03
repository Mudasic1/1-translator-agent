import chainlit as cl
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

GEMINI_API_KEY = "AIzaSyD3J9ttrQ8-IHioZJTucWkgcqZVWSjkTlA"

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

set_tracing_disabled(disabled=True)

def detect_language(text):
    # Simple detection: if contains Urdu characters, treat as Urdu
    for c in text:
        if '\u0600' <= c <= '\u06FF':
            return "ur"
    return "en"

async def translate(agent, text, source_lang, target_lang):
    if source_lang == "en" and target_lang == "ur":
        prompt = f"Translate the following English text to Urdu:\n\n{text}"
    elif source_lang == "ur" and target_lang == "en":
        prompt = f"Translate the following Urdu text to English:\n\n{text}"
    else:
        return "Unsupported language pair."
    result = await Runner.run(agent, prompt)
    return result.final_output

async def chat(agent, user_message, history):
    # Compose a conversation history for the agent
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        if h.get("assistant"):
            messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": user_message})
    prompt = "You are a helpful assistant that can chat in both English and Urdu. If the user asks for translation, translate between English and Urdu. If the user asks who built you, answer 'I was built by Mudasir' in the language of the question. Otherwise, continue the conversation in the language of the user."
    # The OpenAIChatCompletionsModel expects a prompt, so we join the conversation
    full_prompt = prompt + "\n\n"
    for m in messages:
        if m["role"] == "user":
            full_prompt += f"User: {m['content']}\n"
        else:
            full_prompt += f"Assistant: {m['content']}\n"
    result = await Runner.run(agent, full_prompt)
    return result.final_output

async def custom_response(agent, user_message, history):
    # Check for "who built you" intent (case-insensitive, both languages)
    triggers = [
        "who built you", "who made you", "who created you",
        "کس نے آپ کو بنایا", "کس نے تمہیں بنایا", "کس نے بنایا"
    ]
    lowered = user_message.lower()
    if any(trigger in lowered for trigger in triggers):
        # Respond in the language of the question
        if detect_language(user_message) == "ur":
            return "مجھے مدثر نے بنایا ہے"
        else:
            return "I was built by Mudasir"
    # Check for explicit translation request
    if "translate" in lowered or "ترجمہ" in lowered:
        lang = detect_language(user_message)
        if lang == "en":
            return await translate(agent, user_message, "en", "ur")
        elif lang == "ur":
            return await translate(agent, user_message, "ur", "en")
        else:
            return "Sorry, I can only translate between English and Urdu."
    # Otherwise, continue the conversation
    return await chat(agent, user_message, history)

# Store conversation history in memory (per session)
user_histories = {}

@cl.on_message
async def main(message: cl.Message):
    session_id = message.session_id if hasattr(message, "session_id") else "default"
    if session_id not in user_histories:
        user_histories[session_id] = []
    history = user_histories[session_id]

    agent = Agent(
        name="Communicator",
        instructions=(
            "You are a bilingual assistant. You can chat in both English and Urdu, "
            "translate between them, and answer who built you as 'I was built by Mudasir' "
            "in the language of the question. Continue conversations naturally."
        ),
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
    )
    response = await custom_response(agent, message.content, history)
    # Save to history
    history.append({"user": message.content, "assistant": response})
    await cl.Message(content=response).send()