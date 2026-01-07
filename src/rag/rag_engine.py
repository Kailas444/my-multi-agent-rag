import os
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from config.settings import Config

class RAGSystem:
    def __init__(self):
        # We use TF-IDF instead of HuggingFace Embeddings
        self.vectorizer = TfidfVectorizer()
        self.chunks = []
        self._initialize_store()
        
    def _initialize_store(self):
        os.makedirs(Config.FAISS_DIR, exist_ok=True)
        
        # Initialize the index on every startup (it is very fast with TF-IDF)
        print("📄 Initializing Lightweight RAG System...")
        
        if not os.path.exists(Config.PDF_PATH):
            self._create_dummy_pdf()
            
        self._ingest_pdf()

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
        print("📄 Ingesting PDF into TF-IDF Matrix...")
        reader = PdfReader(Config.PDF_PATH)
        
        # Simple text splitting
        self.chunks = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Split text into chunks of 300 chars
                for j in range(0, len(text), 300):
                    self.chunks.append(text[j:j+300])
        
        # Create TF-IDF Matrix (Very fast and memory efficient)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        print("✅ TF-IDF Indexing Complete.")

    def retrieve(self, query: str, k: int = 3) -> Tuple[str, List[Dict]]:
        # Convert query to vector
        query_vec = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Prepare results
        results = []
        for i in top_indices:
            # Filter out very low similarity results (Threshold 0.1)
            if similarities[i] > 0.1:
                results.append((self.chunks[i], {"source": Config.PDF_PATH, "page": 1}))
        
        if not results:
            return "", []
            
        context = "\n\n".join([r[0] for r in results])
        citations = [r[1] for r in results]
        return context, citations
