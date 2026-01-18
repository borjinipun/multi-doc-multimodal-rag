from langchain_text_splitters import CharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents):
    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
