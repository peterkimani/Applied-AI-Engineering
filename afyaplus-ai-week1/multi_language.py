import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "LANGUAGE RULES: Detect the language of the patient message. "
    "ALWAYS respond in that same language. "
    "Supported languages: English, Swahili, Sheng.\n\n"
    "You are an AfyaPlus health assistant. Provide brief, safe guidance. "
    "Never diagnose or prescribe."
)

test_messages = [
    "I have a fever and headache for two days",
    "Nina maumivu ya kichwa kwa siku tatu"  # Swahili: I have a headache for three days
]


for msg in test_messages:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": msg}
        ],
        temperature=0.3,
        max_tokens=200
    )
    print(f"Patient: {msg}")
    print(f"Assistant: {response.choices[0].message.content}\n")