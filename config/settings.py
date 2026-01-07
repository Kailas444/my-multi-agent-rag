import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths (Absolute paths for Docker stability)
    PDF_PATH = "/app/data/knowledge_base.pdf"
    FAISS_DIR = "/app/data/faiss_store"
    
    # Models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
