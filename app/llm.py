from langchain_groq import ChatGroq
from app.config import LLM_MODEL, GROQ_API_KEY
import os

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0
    )