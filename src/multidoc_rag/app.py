from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console

from .config import settings, require_groq_key
from .loader import load_pdfs
from .utils import split_documents
from .vectorstore import build_vectorstore, load_vectorstore
from .llm import LLMClient

app = typer.Typer()
console = Console()


@app.command()
def ingest(
    data_dir: str = typer.Option("data", help="Directory with PDF files"),
    persist_dir: str = typer.Option("doc_db", help="Directory to persist vectorstore"),
):
    """Ingest PDFs from DATA_DIR and build a Chroma vector store."""
    console.print(f"Ingesting PDFs from [bold]{data_dir}[/bold]...")
    docs = load_pdfs(directory=data_dir)
    text_chunks = split_documents(docs)
    build_vectorstore(text_chunks, persist_dir=persist_dir)
    console.print(f"Vectorstore persisted to [bold]{persist_dir}[/bold]")


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask over the documents"),
    persist_dir: str = typer.Option("doc_db", help="Directory where vectorstore was persisted"),
    model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Run a query against the persisted vectorstore and return LLM answer."""
    require_groq_key()
    model = model or settings.model
    temperature = temperature if temperature is not None else settings.temperature

    console.print(f"Loading vectorstore from [bold]{persist_dir}[/bold]...")
    vect = load_vectorstore(persist_dir)
    retriever = vect.as_retriever()
    client = LLMClient(model=model, temperature=temperature)
    answer = client.answer(retriever, question)
    console.rule("Answer")
    console.print(answer)


def main():
    logging.basicConfig(level=logging.INFO)
    app()


if __name__ == "__main__":
    main()