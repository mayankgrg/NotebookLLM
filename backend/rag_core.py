from typing import List
from pathlib import Path
import faiss
import pickle
from openai import OpenAI

client = OpenAI()

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class RAG:
    def __init__(self, vector_store_path="data/faiss_index"):
        self.vector_store_path = Path(vector_store_path)
        self.documents = []
        self.embeddings = None
        self.index = None

    def ingest_documents(self, file_paths: List[str]):
        texts = []
        for file_path in file_paths:
            ext = Path(file_path).suffix
            if ext == ".txt":
                texts.append(Path(file_path).read_text())
            elif ext in [".pdf"]:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                texts.append("\n".join([page.extract_text() for page in reader.pages]))
            # Add DOCX parsing if needed

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(" ".join(texts))
        self.documents.extend(chunks)

        # Create embeddings
        self.embeddings = embedding_model.encode(chunks)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        # Save for later
        faiss.write_index(self.index, str(self.vector_store_path / "index.faiss"))
        with open(self.vector_store_path / "docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def query(self, query_text: str, top_k=5):
        query_vec = embedding_model.encode([query_text])
        D, I = self.index.search(query_vec, top_k)
        results = [self.documents[i] for i in I[0]]
        return results

    def generate_answer(self, query_text: str, top_k=5):
    # Retrieve relevant chunks
        chunks = self.query(query_text, top_k)

        # Generate answer using OpenAI
        prompt = f"Answer the question based on these document chunks:\n\n{chunks}\n\nQuestion: {query_text}\nAnswer:"
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0)
        answer = response.choices[0].message.content.strip()
        return answer