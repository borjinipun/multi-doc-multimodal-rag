from __future__ import annotations

from typing import List

try:
    from langchain_community.document_loaders import (
        UnstructuredPDFLoader,
        DirectoryLoader,
    )
except Exception as exc:  # pragma: no cover - runtime dependency
    UnstructuredPDFLoader = None  # type: ignore
    DirectoryLoader = None  # type: ignore


def load_pdfs(directory: str = "data", glob: str = "*.pdf"):
    """Load PDF documents from `directory` using an UnstructuredPDFLoader.

    Returns a list of Documents (LangChain objects) or raises informative error
    if the optional dependency is not installed.
    """
    if DirectoryLoader is None or UnstructuredPDFLoader is None:
        raise RuntimeError(
            "langchain_community or unstructured is not installed. Please install the extras described in README."
        )
    loader = DirectoryLoader(directory, glob=glob, loader_cls=UnstructuredPDFLoader)
    docs = loader.load()
    return docs
