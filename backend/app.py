from fastapi import FastAPI, UploadFile, File
from typing import List
import shutil
from pathlib import Path

from rag_core import RAG

app = FastAPI()
rag = RAG()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(str(file_path))
    
    rag.ingest_documents(file_paths)
    return {"message": "Files ingested successfully."}

@app.post("/query/")
async def query_text(query: str):
    context_chunks = rag.query(query)
    # Here you can call LLM with context_chunks + query
    from openai import OpenAI
    client = OpenAI()
    context = "\n".join(context_chunks)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful study assistant."},
            {"role": "user", "content": f"Using this context, {context}, answer: {query}"}
        ]
    )
    return {"answer": response.choices[0].message.content}
