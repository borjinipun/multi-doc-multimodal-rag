import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# Chunking
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")
