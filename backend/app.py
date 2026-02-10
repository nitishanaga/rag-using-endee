from fastapi import FastAPI, UploadFile, File
from pdf_loader import extract_text_from_pdf
from embedder import get_embedding
from endee_client import EndeeClient
import os
import shutil

app = FastAPI(
    title="RAG Document Assistant",
    description="RAG-based Semantic Search System using Endee Architecture",
    version="2.0.0"
)

vector_db = EndeeClient()


# -----------------------------
# Utility: Smart Chunking
# -----------------------------
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "RAG Document Assistant using Endee Architecture",
        "status": "running"
    }


# -----------------------------
# Upload PDF Endpoint
# -----------------------------
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    file_location = f"../data/documents/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": f"{file.filename} uploaded successfully"}


# -----------------------------
# Index Document Endpoint
# -----------------------------
@app.post("/index")
def index_document(filename: str):
    file_path = f"../data/documents/{filename}"

    if not os.path.exists(file_path):
        return {"error": "File not found. Please upload first."}

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


# -----------------------------
# Semantic Search Endpoint
# -----------------------------
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


# -----------------------------
# RAG Ask Endpoint (REAL UPGRADE)
# -----------------------------
@app.get("/ask")
def ask_question(question: str):
    if not question.strip():
        return {"error": "Question cannot be empty"}

    # Step 1: Embed question
    query_embedding = get_embedding(question)

    # Step 2: Retrieve top relevant chunks
    retrieved_chunks = vector_db.search(query_embedding, top_k=3)

    if not retrieved_chunks:
        return {"error": "No documents indexed. Please call /index first."}

    # Step 3: Combine context
    context = "\n\n".join(retrieved_chunks)

    # Step 4: Generate structured answer
    answer = f"""
Based on the retrieved document context, here is the relevant information:

{context}

(This response was generated using a Retrieval-Augmented Generation pipeline
powered by vector similarity search.)
"""

    return {
        "question": question,
        "answer": answer
    }

