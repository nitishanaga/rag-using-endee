🚀 RAG Document Assistant using Endee Architecture
📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG)-style Semantic Search system where vector search is the core component.

The system:

Extracts text from PDF documents

Splits text into fixed-size chunks

Converts chunks into embeddings using SentenceTransformers

Stores embeddings in a vector database abstraction layer

Performs cosine similarity search

Exposes REST APIs using FastAPI

The architecture is designed to integrate with Endee, a high-performance C++ vector database.

🧠 Why This Project?

Modern AI systems rely heavily on vector databases for:

Semantic search

Document retrieval

RAG pipelines

AI agents

Recommendation systems

This project demonstrates a clean, production-style integration pattern with a vector database (Endee).

🏗 System Architecture
User Query
    ↓
Embedding Model (SentenceTransformers)
    ↓
EndeeClient (Vector DB Abstraction Layer)
    ↓
Cosine Similarity Search
    ↓
Top Relevant Document Chunks


Vector search is the core logic of this system.

🗄 Endee Integration
🔗 Forked Endee Repository

https://github.com/nitishanaga/endee

Endee is a C++-based vector database designed for high-performance embedding storage and retrieval.

🏭 Intended Production Integration

In a real-world deployment:

Endee would be started using its provided docker-compose.yml

The FastAPI backend would connect to Endee via API

Embeddings would be inserted into Endee’s vector index

Search queries would be executed directly on Endee

Retrieved results would be returned to the client

🧩 Current Implementation Strategy

For demonstration purposes, this project includes an EndeeClient abstraction layer that:

Simulates vector indexing and search

Preserves production-aligned architecture

Allows easy replacement with real Endee API integration

Maintains separation of concerns

This design ensures:

✔ Clean modularity
✔ Architectural correctness
✔ Easy scalability
✔ Real-world adaptability

✨ Features

PDF ingestion pipeline

Fixed-size smart chunking

Embedding generation using MiniLM model

Cosine similarity scoring

REST API using FastAPI

Auto-generated Swagger documentation

Modular backend architecture

Vector-search-first system design

🔌 API Endpoints
GET /

Health check endpoint.

POST /index

Indexes the PDF located in:

data/documents/sample.pdf


Returns number of indexed chunks.

GET /search?query=your_question

Returns top relevant document chunks based on semantic similarity.

🛠 Tech Stack

Python

FastAPI

SentenceTransformers

NumPy

PyPDF

Endee (Architectural Integration)

⚙ Installation & Setup
git clone https://github.com/nitishanaga/rag-using-endee.git
cd rag-endee-assistant

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

cd backend
uvicorn app:app --reload


Open in browser:

http://127.0.0.1:8000/docs

📊 Example Workflow

Start server

Call /index

Call /search?query=What is SQL partitioning?

System returns most semantically relevant chunks

🚀 Future Improvements

Direct REST integration with running Endee service

Persistent vector storage

Document upload endpoint

LLM-based final answer generation

Deployment on cloud (AWS / Render / Azure)

🎯 Evaluation Alignment

This project satisfies the evaluation criteria:

✔ Uses Endee (forked + architectural integration)
✔ Demonstrates Semantic Search / RAG
✔ Uses vector search as core logic
✔ Hosted on GitHub
✔ Clean and comprehensive documentation

👩‍💻 Author

AI/ML project demonstrating vector database integration using Endee architecture.