# lab1_message_structures.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3, max_tokens=300)

messages = [
    SystemMessage(content=(
        "You are a Senior AI Engineer specialising in medical triage interfaces "
        "at AfyaPlus Health in Kenya. Provide professional, safe, structured guidance."
    )),
    HumanMessage(content="Explain the structural difference between a deterministic chain and a dynamic agent.")
]

response = llm.invoke(messages)
print(response.content)