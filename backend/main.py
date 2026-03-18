"""
main.py — FastAPI backend for the Smart Resume–Job Matching System.

Endpoints:
  POST /recommend          — profile-based semantic job search
  POST /resume/analyze     — upload resume PDF → RAG analysis + job matches
  GET  /jobs               — list all indexed jobs
  GET  /health             — health check

Usage:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from endee import Endee

from backend.resume_parser import parse_resume, build_resume_embedding_text
from backend.rag_engine import run_rag_analysis

# ── Constants ─────────────────────────────────────────────────
INDEX_NAME   = "job_listings"
DATA_PATH    = Path(__file__).parent.parent / "data" / "jobs.json"
TOP_K        = 5

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Smart Resume–Job Matching API",
    description="Semantic search + RAG analysis using Endee vector database",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────
print("Loading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to Endee...")
endee_client = Endee()

print("Loading job data...")
all_jobs   = json.loads(DATA_PATH.read_text())
jobs_by_id = {j["id"]: j for j in all_jobs}
print(f"✅ Backend ready — {len(all_jobs)} jobs loaded.")


# ── Schemas ───────────────────────────────────────────────────
class UserProfile(BaseModel):
    name:         Optional[str] = "Candidate"
    skills:       str
    experience:   str
    desired_role: str
    top_k:        Optional[int] = TOP_K


class JobMatch(BaseModel):
    id:               str
    title:            str
    company:          str
    location:         str
    type:             str
    experience:       str
    salary:           str
    skills:           list[str]
    description:      str
    similarity_score: float
    match_reason:     str


class RecommendResponse(BaseModel):
    candidate_name: str
    query_used:     str
    matches:        list[JobMatch]
    total_indexed:  int


class ResumeAnalysisResponse(BaseModel):
    candidate_name:      str
    email:               Optional[str]
    word_count:          int
    resume_text_preview: str
    matches:             list[JobMatch]
    rag_analysis:        dict
    total_indexed:       int


# ── Helpers ───────────────────────────────────────────────────
def build_query_text(profile: UserProfile) -> str:
    return (
        f"I am looking for a {profile.desired_role} position. "
        f"My skills include: {profile.skills}. "
        f"Experience: {profile.experience}."
    )


def generate_match_reason(user_skills: str, job: dict, score: float) -> str:
    user_set = set(s.strip().lower() for s in user_skills.replace(",", " ").split())
    job_set  = set(s.strip().lower() for s in job.get("skills", []))
    matched  = user_set & job_set
    pct      = round(score * 100, 1)
    if matched:
        return f"{pct}% semantic match — shared skills: {', '.join(sorted(matched)[:3])}"
    return f"{pct}% semantic match based on role and experience"


def query_endee(query_vector: list, top_k: int) -> list:
    try:
        index = endee_client.get_index(name=INDEX_NAME)
        return index.query(vector=query_vector, top_k=top_k)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Endee query failed: {str(e)}. Ensure Endee is running and indexer was executed."
        )


def build_matches(results, user_skills: str) -> list[JobMatch]:
    out = []
    for r in results:
        job = jobs_by_id.get(r.id)
        if not job:
            continue
        out.append(JobMatch(
            id=r.id,
            title=job["title"],
            company=job["company"],
            location=job["location"],
            type=job["type"],
            experience=job["experience"],
            salary=job["salary"],
            skills=job["skills"],
            description=job["description"],
            similarity_score=round(float(r.similarity), 4),
            match_reason=generate_match_reason(user_skills, job, float(r.similarity)),
        ))
    return out


# ── Routes ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "all-MiniLM-L6-v2",
            "endee_index": INDEX_NAME, "jobs_loaded": len(all_jobs)}


@app.get("/jobs")
def list_jobs():
    return {"total": len(all_jobs), "jobs": all_jobs}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(profile: UserProfile):
    """Semantic job search from manual profile input."""
    query_text   = build_query_text(profile)
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    results      = query_endee(query_vector, profile.top_k or TOP_K)
    if not results:
        raise HTTPException(status_code=404, detail="No matches found.")
    return RecommendResponse(
        candidate_name=profile.name or "Candidate",
        query_used=query_text,
        matches=build_matches(results, profile.skills),
        total_indexed=len(all_jobs),
    )


@app.post("/resume/analyze", response_model=ResumeAnalysisResponse)
async def analyze_resume(
    file:         UploadFile = File(...),
    top_k:        int        = Form(default=5),
    groq_api_key: str        = Form(default=""),
):
    """
    RAG Resume Analysis Pipeline:
      1. Parse PDF resume → extract text + sections
      2. Embed resume via sentence-transformer
      3. Query Endee → retrieve top-K semantically similar jobs
      4. Run RAG: resume + retrieved jobs → LLM analysis
      5. Return matches + deep AI insight
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 5MB.")

    try:
        parsed = parse_resume(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if parsed["word_count"] < 30:
        raise HTTPException(
            status_code=422,
            detail="Resume is empty or unreadable. Upload a text-based PDF."
        )

    embed_text   = build_resume_embedding_text(parsed)
    query_vector = model.encode(embed_text, normalize_embeddings=True).tolist()

    results = query_endee(query_vector, top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No job matches found.")

    matches      = build_matches(results, parsed["raw_text"][:500])
    matches_dict = [m.model_dump() for m in matches]

    rag_analysis = run_rag_analysis(
        resume_text=parsed["raw_text"],
        matched_jobs=matches_dict,
        groq_api_key=groq_api_key or None,
    )

    return ResumeAnalysisResponse(
        candidate_name=parsed["name"] or "Candidate",
        email=parsed["email"],
        word_count=parsed["word_count"],
        resume_text_preview=parsed["raw_text"][:500],
        matches=matches,
        rag_analysis=rag_analysis,
        total_indexed=len(all_jobs),
    )
