from langchain_chroma import Chroma
from app.config import VECTORSTORE_DIR
from app.embeddings import get_embedding_model

def build_vectorstore(documents):
    embedding = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=VECTORSTORE_DIR
    )
    return vectorstore


def load_vectorstore():
    embedding = get_embedding_model()
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embedding
    )
