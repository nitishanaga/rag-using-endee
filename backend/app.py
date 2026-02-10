from fastapi import FastAPI
from pdf_loader import extract_text_from_pdf
from embedder import get_embedding
from endee_client import EndeeClient
import os

app = FastAPI(
    title="RAG Document Assistant",
    description="Semantic Search System using Endee Architecture",
    version="1.0.0"
)

vector_db = EndeeClient()


# ----------- Utility: Smart Chunking -----------
def chunk_text(text, chunk_size=500):
    """
    Splits text into fixed-size chunks.
    More professional than simple sentence splitting.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


# ----------- Root Endpoint -----------
@app.get("/")
def home():
    return {
        "message": "RAG Document Assistant using Endee Architecture",
        "status": "running"
    }


# ----------- Index Document -----------
@app.post("/index")
def index_document():
    file_path = "../data/documents/sample.pdf"

    if not os.path.exists(file_path):
        return {"error": "PDF not found in data/documents folder"}

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        return {"error": "PDF contains no readable text"}

    chunks = chunk_text(text)

    for chunk in chunks:
        if chunk.strip():
            embedding = get_embedding(chunk)
            vector_db.add_document(embedding, chunk)

    return {
        "message": "Document indexed successfully",
        "total_chunks": len(chunks)
    }


# ----------- Search Endpoint -----------
@app.get("/search")
def search(query: str):
    if not query.strip():
        return {"error": "Query cannot be empty"}

    query_embedding = get_embedding(query)
    results = vector_db.search(query_embedding)

    return {
        "query": query,
        "top_results": results
    }
