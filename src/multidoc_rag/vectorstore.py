from __future__ import annotations

from typing import Iterable

try:
    from langchain_chroma import Chroma
except Exception:
    Chroma = None  # type: ignore

from .embeddings import get_embedding


def build_vectorstore(docs: Iterable, persist_dir: str = "doc_db"):
    if Chroma is None:
        raise RuntimeError("langchain-chroma not installed")
    embedding = get_embedding()
    vect = Chroma.from_documents(documents=list(docs), embedding=embedding, persist_directory=persist_dir)
    return vect


def load_vectorstore(persist_dir: str = "doc_db"):
    if Chroma is None:
        raise RuntimeError("langchain-chroma not installed")
    embedding = get_embedding()
    vect = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    return vect
