import os
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import Config

class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vector_store = self._initialize_store()
        
    def _initialize_store(self) -> FAISS:
        # Ensure directory exists
        os.makedirs(Config.FAISS_DIR, exist_ok=True)
        
        if os.path.exists(os.path.join(Config.FAISS_DIR, "index.faiss")):
            print("✅ Loading existing FAISS index...")
            return FAISS.load_local(Config.FAISS_DIR, self.embeddings, allow_dangerous_deserialization=True)
        
        print(f"⚠️ Creating dummy PDF at {Config.PDF_PATH}...")
        self._create_dummy_pdf()
        self._ingest_pdf()
        return self.vector_store

    def _create_dummy_pdf(self):
        from fpdf import FPDF
        os.makedirs(os.path.dirname(Config.PDF_PATH), exist_ok=True)
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        text = "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines. Machine Learning (ML) is a subset of AI focused on building systems that learn from data. Natural Language Processing (NLP) enables computers to understand human language."
        for line in text.split('. '):
            pdf.cell(0, 10, line + '.', 0, 1)
        pdf.output(Config.PDF_PATH)

    def _ingest_pdf(self):
        from pypdf import PdfReader
        print("📄 Ingesting PDF...")
        reader = PdfReader(Config.PDF_PATH)
        docs = [Document(page_content=p.extract_text(), metadata={"source": "knowledge_base.pdf", "page": i+1}) 
                for i, p in enumerate(reader.pages) if p.extract_text()]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(Config.FAISS_DIR)
        print("✅ FAISS Index Saved.")

    def retrieve(self, query: str, k: int = 3) -> Tuple[str, List[Dict]]:
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        relevant_docs = [d for d, score in docs if score < 1.4] 
        
        if not relevant_docs:
            return "", []
            
        context = "\n\n".join([d.page_content for d in relevant_docs])
        citations = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in relevant_docs]
        return context, citations
