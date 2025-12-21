from __future__ import annotations

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None  # type: ignore


def get_embedding():
    if HuggingFaceEmbeddings is None:
        raise RuntimeError("langchain-huggingface not installed")
    return HuggingFaceEmbeddings()
