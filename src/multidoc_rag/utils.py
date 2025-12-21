from __future__ import annotations

from typing import Iterable, List

try:
    from langchain_text_splitters import CharacterTextSplitter
except Exception:
    CharacterTextSplitter = None  # type: ignore


def split_documents(docs: Iterable, chunk_size: int = 2000, chunk_overlap: int = 500):
    """Split LangChain Document-like objects into text chunks."""
    if CharacterTextSplitter is None:
        raise RuntimeError("langchain-text-splitters is not installed")
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(list(docs))
