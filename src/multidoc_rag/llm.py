from __future__ import annotations

from typing import Iterable

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None  # type: ignore

from .prompt import make_prompt


class LLMClient:
    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.0):
        if ChatGroq is None:
            raise RuntimeError("langchain-groq not installed")
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.prompt = make_prompt()

    def answer(self, retriever, question: str, top_k: int = 4) -> str:
        # Prefer retriever.invoke if present (some retrievers have invoke)
        if hasattr(retriever, "invoke"):
            ctx = retriever.invoke(question)
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
            ctx = "\n".join([d.page_content for d in docs])
        else:
            raise RuntimeError("Retriever does not support invoke or get_relevant_documents")

        formatted = self.prompt.format_messages(context=ctx, input=question)
        ai_msg = self.llm.invoke(formatted)
        return getattr(ai_msg, "content", str(ai_msg))
