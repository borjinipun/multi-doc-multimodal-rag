from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    DirectoryLoader
)

from app.config import DATA_DIR

def load_documents():
    loader = DirectoryLoader(
        DATA_DIR,
        glob="*.pdf",
        loader_cls=UnstructuredPDFLoader
    )
    return loader.load()
