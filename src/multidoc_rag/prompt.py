from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)


def make_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
