import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
cloud_client = OpenAI()
local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

SYSTEM_PROMPT = "You are a health assistant. Provide brief, safe guidance."
patient_message = "I have chest pain when I breathe deeply"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": patient_message}
]

start = time.time()
cloud_response = cloud_client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, temperature=0.3)
cloud_time = time.time() - start
start = time.time()
local_response = local_client.chat.completions.create(
    model="llama3.2", messages=messages, temperature=0.3)
local_time = time.time() - start
print(f"{'Metric':<20} {'GPT-4o-mini':<25} {'Llama 3.2 (local)':<25}")
print("-" * 70)
print(f"Response time:       {cloud_time:.2f}s                     {local_time:.2f}s")
print(f"Response length:     {len(cloud_response.choices[0].message.content)} chars             {len(local_response.choices[0].message.content)} chars")
print(f"Cloud preview: {cloud_response.choices[0].message.content[:200]}...")
print(f"Local preview: {local_response.choices[0].message.content[:200]}...")