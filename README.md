# ✦ Smart Resume–Job Matching System
### Powered by Endee Vector Database + RAG

> **Tap Academy × Endee.io Assignment** — A production-grade AI application combining semantic vector search and Retrieval-Augmented Generation (RAG) for intelligent career matching.

---

## 📌 What This Project Does

This system has **two AI-powered modes**:

| Mode | Description |
|---|---|
| **Profile Search** | Type your skills and role → Endee finds the best matching jobs semantically |
| **Resume Analyzer** | Upload your PDF resume → Endee retrieves matches → LLM generates deep analysis |

The Resume Analyzer is the flagship feature. It parses your resume, embeds it, retrieves the top semantically matching jobs from Endee, then passes the resume + retrieved jobs to an LLM (via Groq) to generate a detailed **AI Analysis Report** including fit verdicts, skills gap analysis, resume improvement tips, and a candidacy score.

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║              INDEXING PIPELINE (run once)                    ║
║                                                              ║
║  jobs.json → Text Builder → Sentence Transformer (384D)     ║
║                                      ↓                       ║
║                              Endee DB (upsert)               ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║           PROFILE SEARCH PIPELINE                            ║
║                                                              ║
║  User Input → Sentence Transformer → Query Vector           ║
║                                           ↓                  ║
║                                    Endee cosine search       ║
║                                           ↓                  ║
║                                  Ranked Job Matches          ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║           RAG RESUME ANALYSIS PIPELINE                       ║
║                                                              ║
║  PDF Upload → PyMuPDF Parser → Section Extractor            ║
║                                      ↓                       ║
║                          Sentence Transformer (384D)         ║
║                                      ↓                       ║
║                          Endee cosine search (RETRIEVE)      ║
║                                      ↓                       ║
║            Resume Text + Retrieved Jobs = RAG Context        ║
║                                      ↓                       ║
║                      Groq LLM (llama3-8b-8192) (GENERATE)   ║
║                                      ↓                       ║
║          ┌──────────────────────────────────────────┐        ║
║          │  • Candidacy Score (0–100)               │        ║
║          │  • Per-job fit verdict                   │        ║
║          │  • Why it's a good/bad fit               │        ║
║          │  • Skills gap report                     │        ║
║          │  • Resume improvement suggestions        │        ║
║          │  • Interview tips per role               │        ║
║          │  • Next steps action plan                │        ║
║          └──────────────────────────────────────────┘        ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 🗄️ How Endee is Used (Core)

Endee is **not a side component** — it is the search engine that powers both features:

```python
# 1. Create index
client.create_index(
    name="job_listings",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8,
)

# 2. Index all jobs as vectors
index.upsert([{"id": job_id, "vector": embedding, "meta": {...}}])

# 3. Query with user/resume vector
results = index.query(vector=query_vector, top_k=5)
# → Returns ranked job IDs + similarity scores
```

Key Endee features used:
- **HNSW indexing** — sub-millisecond approximate nearest neighbour search
- **Cosine similarity** — best for normalized semantic embeddings
- **INT8 precision** — 4x memory reduction with minimal accuracy loss
- **Metadata storage** — job metadata stored and retrieved alongside vectors

---

## 📁 Project Structure

```
├── docker-compose.yml          # Starts Endee vector DB
├── requirements.txt            # Python dependencies
├── data/
│   └── jobs.json               # 20 Indian tech job listings
├── backend/
│   ├── __init__.py
│   ├── indexer.py              # Embeds & loads jobs into Endee
│   ├── main.py                 # FastAPI server (2 endpoints)
│   ├── resume_parser.py        # PDF parsing + section extraction
│   └── rag_engine.py           # RAG: Groq LLM integration + fallback
└── frontend/
    └── app.py                  # Streamlit UI (2 modes)
```

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.10+
- Docker Desktop (running)
- (Optional) Free Groq API key from console.groq.com

### Step 1 — Start Endee
```bash
docker compose up -d
curl http://localhost:8080/api/v1/index/list  # verify ✓
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Index jobs into Endee
```bash
python backend/indexer.py
```

### Step 4 — Start API (Terminal 1)
```bash
uvicorn backend.main:app --reload --port 8000
```

### Step 5 — Start UI (Terminal 2)
```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501** 🎉

---

## 🔌 API Reference

### `POST /recommend` — Profile-based job search
```json
{
  "name": "Arjun Sharma",
  "desired_role": "ML Engineer",
  "skills": "Python, PyTorch, NLP",
  "experience": "2 years building classification models",
  "top_k": 5
}
```

### `POST /resume/analyze` — RAG resume analysis
Multipart form:
- `file`: PDF resume (max 5MB)
- `top_k`: number of matches (default 5)
- `groq_api_key`: optional Groq key for LLM analysis

### `GET /jobs` — List all indexed jobs
### `GET /health` — Health check

---

## 🎯 Tech Stack

| Layer | Technology |
|---|---|
| Vector Database | **Endee** (Docker, localhost:8080) |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` (384D) |
| PDF Parsing | PyMuPDF (fitz) |
| RAG LLM | Groq API — `llama3-8b-8192` |
| Backend | FastAPI + Pydantic |
| Frontend | Streamlit (custom CSS) |
| Containerization | Docker Compose |

---

## 📄 License
Apache 2.0
