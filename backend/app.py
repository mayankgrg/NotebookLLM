import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from models import ChatRequest, ChatResponse, IngestResponse, DocMeta
from rag_core import store, answer_query

app = FastAPI(title="Notebook-LLM-lite", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    payload = []
    for f in files:
        data = await f.read()
        payload.append((data, f.filename))
    docs = store.ingest_files(payload)
    return {"docs": docs}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    answer, citations = answer_query(req.query, top_k=req.top_k, max_sentences=req.max_sentences)
    return {"answer": answer, "citations": citations}