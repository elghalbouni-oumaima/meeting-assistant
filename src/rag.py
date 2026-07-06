"""
rag.py
RAG pipeline for Q&A over the meeting transcript.

Pipeline:
  1. Chunk the transcript into overlapping pieces
  2. Embed each chunk with sentence-transformers
  3. Build a FAISS index for fast similarity search
  4. At query time: embed the question, retrieve top-k chunks
  5. Build an augmented prompt and generate the answer with flan-t5
"""

import re
import yaml
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAG_CONFIG = CONFIG["rag"]

# module-level model variables — loaded once
_embedder  = None
_tokenizer = None
_model     = None


def _load_embedder():
    global _embedder
    if _embedder is None:
        print(f"DEBUG — loading embedder: {RAG_CONFIG['embedder']}")
        _embedder = SentenceTransformer(RAG_CONFIG["embedder"])


def _load_generator():
    global _tokenizer, _model
    if _tokenizer is None:
        model_name = CONFIG["models"]["action_extractor"]  # reuse flan-t5-large
        print(f"DEBUG — loading RAG generator: {model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model     = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)


# ── chunking ───────────────────────────────────────────────────────────────────

def _clean_transcript(transcript: str) -> str:
    """Remove metadata lines before chunking."""
    lines = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(line.lower().startswith(k) for k in
               ["meeting:", "attendees:", "date:", "location:"]):
            continue
        lines.append(line)
    return "\n".join(lines)


def _chunk(transcript: str) -> list[dict]:
    """
    Splits the transcript into overlapping chunks.
    Each chunk keeps full speaker turns together where possible.
    Returns: [{"text": str, "index": int}, ...]
    """
    chunk_size = RAG_CONFIG["chunk_size"]
    overlap    = RAG_CONFIG["chunk_overlap"]
    clean      = _clean_transcript(transcript)
    words      = clean.split()
    chunks     = []
    start      = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({"text": chunk, "index": len(chunks)})
        start += chunk_size - overlap

    print(f"DEBUG — created {len(chunks)} chunks")
    return chunks


# ── indexing ───────────────────────────────────────────────────────────────────

def build_index(transcript: str) -> tuple:
    """
    Embeds all chunks and builds a FAISS index.
    Returns (chunks, index) — store both in st.session_state.
    """
    _load_embedder()
    chunks  = _chunk(transcript)
    texts   = [c["text"] for c in chunks]
    vectors = _embedder.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")

    dim   = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    print(f"DEBUG — FAISS index built: {index.ntotal} vectors, dim={dim}")
    return chunks, index


# ── retrieval ──────────────────────────────────────────────────────────────────

def retrieve(query: str, chunks: list, index, top_k: int = 3) -> list[str]:
    """
    Encodes the query and returns the top-k most relevant chunks.
    """
    _load_embedder()
    query_vec = _embedder.encode(
        [query], convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_vec, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(chunks):
            results.append({
                "text"    : chunks[idx]["text"],
                "distance": float(dist),
            })

    print(f"DEBUG — retrieved {len(results)} chunks for: '{query}'")
    return results


# ── generation ─────────────────────────────────────────────────────────────────

def answer(query: str, chunks: list, index) -> dict:
    """
    Full RAG pipeline — retrieve + generate.

    Returns:
    {
        "answer" : str,
        "context": list[str]   ← the retrieved chunks shown to user
    }
    """
    _load_generator()

    retrieved = retrieve(query, chunks, index, top_k=RAG_CONFIG["top_k"])
    context   = "\n".join([r["text"] for r in retrieved])

    prompt = f"""Answer the question using only the meeting transcript below.
If the answer is not in the transcript, say exactly: "This was not discussed in the meeting."

Transcript:
{context}

Question: {query}

Answer:"""

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    output = _model.generate(
        **inputs,
        max_new_tokens=RAG_CONFIG["max_new_tokens"],
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )

    response = _tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"DEBUG — RAG answer: {response}")

    return {
        "answer" : response,
        "context": [r["text"] for r in retrieved],
    }