from fastapi import FastAPI, UploadFile, File, Body
from typing import List
import shutil
from pathlib import Path
from pydantic import BaseModel


from rag_core import RAG

app = FastAPI()
rag = RAG()

class Query(BaseModel):
    query: str

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

# @app.post("/query/")
# async def query_text(query: Query):
#     # context_chunks = rag.query(query)
#     # Here you can call LLM with context_chunks + query
#     context_chunks = rag.query(query.query)
#     answer = llm.generate_answer(query.query, context_chunks)
#     return {"answer": answer}

@app.post("/query/")
async def query_text(query: Query):
    try:
        answer = rag.generate_answer(query.query)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}



