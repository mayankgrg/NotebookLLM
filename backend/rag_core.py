# backend/rag_core.py
from typing import List
from pathlib import Path
import os
import re
import faiss
import pickle
from openai import OpenAI
import os

api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key 7643659435: {api_key}")
client = OpenAI(api_key=api_key)

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import OpenAI

# Make sure FAISS storage folders exist
DEFAULT_STORE = Path("data/faiss_index")
DEFAULT_STORE.mkdir(parents=True, exist_ok=True)

# Load model for embeddings (small & fast)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenAI client (expects OPENAI_API_KEY in env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _normalize_text(s: str) -> str:
    """Replace curly quotes and ensure we don't choke on odd chars."""
    if not s:
        return ""
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    # Force to utf-8-clean string (strip any odd leftovers)
    return s.encode("utf-8", errors="ignore").decode("utf-8")

class RAG:
    def __init__(self, vector_store_path: str = str(DEFAULT_STORE)):
        self.vector_store_path = Path(vector_store_path)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.documents: List[str] = []
        self.embeddings = None
        self.index = None

    def ingest_documents(self, file_paths: List[str]):
        texts = []
        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()

            if ext == ".txt":
                text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                texts.append(_normalize_text(text))

            elif ext == ".pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                pages = []
                for page in reader.pages:
                    pages.append((page.extract_text() or ""))
                text = "\n".join(pages)
                texts.append(_normalize_text(text))

            # elif ext == ".docx":  # optional
            #     import docx
            #     doc = docx.Document(file_path)
            #     text = "\n".join([p.text for p in doc.paragraphs])
            #     texts.append(_normalize_text(text))

        if not texts:
            raise ValueError("No supported files found to ingest.")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(" ".join(texts))
        chunks = [c.strip() for c in chunks if c.strip()]

        if not chunks:
            raise ValueError("No text could be extracted from the uploaded files.")

        self.documents = chunks

        # Create embeddings & FAISS index
        self.embeddings = embedding_model.encode(self.documents)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        # Persist index + docs
        faiss.write_index(self.index, str(self.vector_store_path / "index.faiss"))
        with open(self.vector_store_path / "docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def _ensure_loaded(self):
        """Lazy-load index/docs from disk if not in memory."""
        if self.index is not None and self.documents:
            return
        index_path = self.vector_store_path / "index.faiss"
        docs_path = self.vector_store_path / "docs.pkl"
        if not index_path.exists() or not docs_path.exists():
            raise ValueError("No documents ingested yet. Please upload materials first.")
        self.index = faiss.read_index(str(index_path))
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

    def query(self, query_text: str, top_k=5) -> List[str]:
        self._ensure_loaded()
        q = _normalize_text(query_text or "")
        if not q:
            return []
        q_vec = embedding_model.encode([q])
        D, I = self.index.search(q_vec, top_k)
        results = []
        for i in I[0]:
            if 0 <= i < len(self.documents):
                results.append(self.documents[i])
        return results

    def generate_answer(self, question: str, top_k=5) -> str:
        self._ensure_loaded()
        q = _normalize_text(question)
        if not q:
            return "Please provide a question."

        contexts = self.query(q, top_k=top_k)
        if not contexts:
            return "I couldn't retrieve any relevant context. Try re-ingesting your files."

        # Build prompt
        context_block = "\n\n".join(contexts[:top_k])
        system_msg = (
            "You are a helpful study assistant. Answer ONLY using the provided course "
            "materials. If the answer is not in the materials, say you don't know."
        )
        user_msg = f"Question: {q}\n\nContext:\n{context_block}\n\nAnswer as clearly as possible."

        # OpenAI chat (new client)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # or another chat-capable model you have access to
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content or ""
        return _normalize_text(answer).strip()
