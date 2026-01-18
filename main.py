from app.loaders import load_documents
from app.splitter import split_documents
from app.vectorstore import build_vectorstore, load_vectorstore
from app.retriever import get_retriever
from app.rag_pipeline import RAGPipeline
import os

def setup_vectorstore():
    if not os.path.exists("vectorstore"):
        print("Building vectorstore...")
        docs = load_documents()
        chunks = split_documents(docs)
        return build_vectorstore(chunks)
    else:
        print("Loading existing vectorstore...")
        return load_vectorstore()

def main():
    vectorstore = setup_vectorstore()
    retriever = get_retriever(vectorstore)
    rag = RAGPipeline(retriever)

    queries = [
        "What is the filling pressure of Potable Water System on A320?",
        "What is the approximate height of the potable water service panel from the ground?",
        "What is the usable capacity of the A320 potable water tank?"
    ]

    for q in queries:
        print(f"\nQ: {q}")
        print(f"A: {rag.answer(q)}")

if __name__ == "__main__":
    main()
