# lab2_chat_history.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.2)

chat_history = [
    SystemMessage(content="You are an AfyaPlus clinic scheduling assistant in Kisumu."),
    HumanMessage(content="Hi, do you have slots open tomorrow morning?")
]

first_reply = llm.invoke(chat_history)
print(f"AI: {first_reply.content}\n")

chat_history.append(AIMessage(content=first_reply.content))
chat_history.append(HumanMessage(content="Great, lock in the 9:00 AM slot for me please."))

second_reply = llm.invoke(chat_history)
print(f"AI: {second_reply.content}")