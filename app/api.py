from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import shutil
from pathlib import Path
import os

from app.loaders import load_documents
from app.splitter import split_documents
from app.vectorstore import build_vectorstore, load_vectorstore
from app.retriever import get_retriever
from app.rag_pipeline import RAGPipeline

DATA_DIR = Path("data")

app = FastAPI(
    title="Multi-Document RAG API",
    version="1.0.0"
)

# ---------
# Globals (loaded once)
# ---------
rag_pipeline: RAGPipeline | None = None


# ---------
# Schemas
# ---------
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


# ---------
# Startup logic
# ---------
@app.on_event("startup")
def startup_event():
    global rag_pipeline

    if os.path.exists("vectorstore"):
        vectorstore = load_vectorstore()
    else:
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = build_vectorstore(chunks)

    retriever = get_retriever(vectorstore)
    rag_pipeline = RAGPipeline(retriever)


# ---------
# Health check
# ---------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------
# Query endpoint
# ---------
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")

    answer = rag_pipeline.answer(request.question)
    return {"answer": answer}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs allowed")

    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Rebuild vectorstore
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)

    global rag_pipeline
    rag_pipeline = RAGPipeline(get_retriever(vectorstore))

    return {"status": "uploaded", "filename": file.filename}