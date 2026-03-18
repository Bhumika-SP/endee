"""
indexer.py — Loads job listings into Endee vector database.
Run this ONCE before starting the API server.

Usage:
    python indexer.py

Make sure Endee is running on localhost:8080 via Docker:
    docker compose up -d
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

INDEX_NAME = "job_listings"
DIMENSION = 384  # all-MiniLM-L6-v2 output dimension
DATA_PATH = Path(__file__).parent.parent / "data" / "jobs.json"


def build_job_text(job: dict) -> str:
    """Combine job fields into a rich text for embedding."""
    skills_str = ", ".join(job.get("skills", []))
    return (
        f"Job title: {job['title']}. "
        f"Company: {job['company']}. "
        f"Location: {job['location']}. "
        f"Type: {job['type']}. "
        f"Experience required: {job['experience']}. "
        f"Key skills: {skills_str}. "
        f"Description: {job['description']}"
    )


def main():
    print("=" * 55)
    print("  AI Job Recommender — Endee Indexer")
    print("=" * 55)

    # 1. Load embedding model
    print("\n[1/4] Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("      ✓ Model loaded")

    # 2. Connect to Endee
    print("\n[2/4] Connecting to Endee at localhost:8080...")
    client = Endee()
    print("      ✓ Connected to Endee")

    # 3. Create index (drop if exists for clean re-indexing)
    print(f"\n[3/4] Setting up index '{INDEX_NAME}'...")
    try:
        existing = [idx.name for idx in client.list_indexes()]
        if INDEX_NAME in existing:
            client.delete_index(INDEX_NAME)
            print(f"      ↺  Dropped existing index '{INDEX_NAME}'")
    except Exception:
        pass

    client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        space_type="cosine",
        precision=Precision.INT8,
    )
    print(f"      ✓ Index '{INDEX_NAME}' created (dim={DIMENSION}, cosine, INT8)")

    # 4. Embed and upsert jobs
    print("\n[4/4] Embedding and indexing job listings...")
    jobs = json.loads(DATA_PATH.read_text())
    index = client.get_index(name=INDEX_NAME)

    texts = [build_job_text(j) for j in jobs]
    print(f"      Generating embeddings for {len(jobs)} jobs...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    vectors = []
    for job, embedding in zip(jobs, embeddings):
        vectors.append({
            "id": job["id"],
            "vector": embedding.tolist(),
            "meta": {
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "type": job["type"],
                "experience": job["experience"],
                "salary": job["salary"],
                "skills": ", ".join(job["skills"]),
                "description": job["description"][:300],  # truncate for meta
            },
        })

    index.upsert(vectors)
    time.sleep(0.5)  # allow Endee to flush

    print(f"\n{'=' * 55}")
    print(f"  ✅ Successfully indexed {len(jobs)} jobs into Endee!")
    print(f"  Index: '{INDEX_NAME}' | Dimension: {DIMENSION}")
    print(f"{'=' * 55}")
    print("\n  ▶  Now run the API:  uvicorn backend.main:app --reload")
    print()


if __name__ == "__main__":
    main()
