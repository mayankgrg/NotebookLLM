from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    use_llm: bool = False
    max_sentences: int = 6

class ChatResponse(BaseModel):
    answer: str
    citations: List[dict]

class DocMeta(BaseModel):
    id: str
    filename: str
    n_chunks: int

class IngestResponse(BaseModel):
    docs: List[DocMeta]