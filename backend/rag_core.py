import os
import io
import json
import uuid
import pickle
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import faiss

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'storage')
INDEX_PATH = os.path.join(STORAGE_DIR, 'faiss.index')
META_PATH = os.path.join(STORAGE_DIR, 'meta.pkl')
CHUNK_SIZE = 700  # characters
CHUNK_OVERLAP = 120

MODEL_NAME = os.environ.get('EMBED_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

class Store:
    def __init__(self):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.meta = []  # list of dicts: {id, filename, chunk, text}
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self._load()

    def _load(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, 'rb') as f:
            self.meta = pickle.load(f)

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, 'wb') as f:
            pickle.dump(self.meta, f)

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + CHUNK_SIZE)
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
            if start < 0:
                start = 0
        return chunks

    def _read_file(self, file_bytes: bytes, filename: str) -> str:
        name = filename.lower()
        try:
            if name.endswith(('.txt', '.md', '.csv', '.py', '.log')):
                return file_bytes.decode('utf-8', errors='ignore')
            elif name.endswith('.json'):
                return json.dumps(json.loads(file_bytes.decode('utf-8', errors='ignore')), indent=2)
            elif name.endswith('.pdf'):
                # Light-weight PDF text extraction using PyPDF2 if available.
                try:
                    import PyPDF2
                except Exception:
                    return ''
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                pages = []
                for p in reader.pages:
                    pages.append(p.extract_text() or '')
                return '\n'.join(pages)
            else:
                return file_bytes.decode('utf-8', errors='ignore')
        except Exception:
            return ''

    def ingest_files(self, files: List[Tuple[bytes, str]]) -> List[Dict]:
        # Prepare data
        records = []
        for data, fname in files:
            text = self._read_file(data, fname)
            if not text.strip():
                continue
            chunks = self._chunk_text(text)
            for i, ch in enumerate(chunks):
                records.append({
                    'id': str(uuid.uuid4()),
                    'filename': fname,
                    'chunk': i,
                    'text': ch
                })
        if not records:
            return []

        # Embed
        texts = [r['text'] for r in records]
        embs = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embs = np.array(embs).astype('float32')

        # Build or add to FAISS index
        d = embs.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine sim
        self.index.add(embs)

        # Extend metadata and persist
        start_len = len(self.meta)
        self.meta.extend(records)
        self._save()

        # Return doc-level stats
        by_file: Dict[str, int] = {}
        for r in records:
            by_file[r['filename']] = by_file.get(r['filename'], 0) + 1
        out = [{'id': f'{fname}', 'filename': fname, 'n_chunks': n} for fname, n in by_file.items()]
        return out

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or not self.meta:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(q).astype('float32'), top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            hits.append({
                'score': float(score),
                'text': m['text'],
                'filename': m['filename'],
                'chunk': m['chunk']
            })
        return hits

store = Store()

# ---- Summarization (extractive, no LLM required) ----

def extractive_summary(passages: List[str], max_sentences: int = 6) -> str:
    """Build a concise summary from relevant passages using TF-IDF + sentence scoring."""
    if not passages:
        return "No relevant content found in the ingested documents."

    # Split into sentences, keep mapping
    sents = []
    for p in passages:
        for s in sent_tokenize(p):
            s_clean = s.strip()
            if len(s_clean) > 20:
                sents.append(s_clean)
    if not sents:
        return passages[0][:400]

    # TF-IDF scoring against the pseudo-document (all sentences)
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), stop_words='english')
    X = vectorizer.fit_transform(sents)
    doc_center = X.mean(axis=0)
    sims = cosine_similarity(X, doc_center)
    scored = list(zip([float(x) for x in sims.ravel()], sents))

    # Deduplicate similar sentences (MMR-lite)
    picked = []
    used = []
    for score, sent in sorted(scored, key=lambda x: x[0], reverse=True):
        if len(picked) >= max_sentences:
            break
        ok = True
        for u in used:
            # Jaccard on words as cheap proxy
            a, b = set(sent.lower().split()), set(u.lower().split())
            j = len(a & b) / (len(a | b) + 1e-9)
            if j > 0.6:
                ok = False
                break
        if ok:
            picked.append(sent)
            used.append(sent)
    return " ".join(picked)


def answer_query(query: str, top_k: int = 5, max_sentences: int = 6) -> Tuple[str, List[Dict]]:
    hits = store.retrieve(query, top_k=top_k)
    passages = [h['text'] for h in hits]
    summary = extractive_summary(passages, max_sentences=max_sentences)
    citations = [{
        'filename': h['filename'],
        'chunk': h['chunk'],
        'score': h['score']
    } for h in hits]
    return summary, citations