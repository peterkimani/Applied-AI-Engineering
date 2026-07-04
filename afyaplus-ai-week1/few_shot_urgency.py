import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def analyse_patient_urgency_few_shot(new_patient_query):
    """The prompt engineering happens in the messages list below.
    The system message defines the task; the user/assistant pairs
    demonstrate the EXACT output format; the final user message is
    the live target the model will classify in the same style."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            # ---- ROLE & TASK ----
            {"role": "system",
             "content": "Classify incoming user medical queries into exactly "
                        "one category: CRITICAL, NON_URGENT, or ROUTINE."},
            # ---- FEW-SHOT EXAMPLE 1: a CRITICAL case ----
            {"role": "user",
             "content": "Query: I cannot breathe and my left arm feels numb."},
            {"role": "assistant",
             "content": "Category: CRITICAL"},
            # ---- FEW-SHOT EXAMPLE 2: a ROUTINE case ----
            {"role": "user",
             "content": "Query: I need to renew my allergy pills prescription "
                        "next month."},
            {"role": "assistant",
             "content": "Category: ROUTINE"},
            # ---- LIVE TARGET QUERY ----
            {"role": "user",
             "content": f"Query: {new_patient_query}"}
        ]
    )
    return response.choices[0].message.content

verdict = analyse_patient_urgency_few_shot(
    "My child has a mild fever but is laughing and playing.")
print(verdict)  # Expected: Category: NON_URGENT