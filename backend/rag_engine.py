"""
rag_engine.py — RAG (Retrieval Augmented Generation) engine.

Flow:
  1. Resume text + top-K matched jobs (retrieved from Endee) = context
  2. Context is sent to Groq LLM (llama3-8b-8192, free tier)
  3. LLM generates:
     - Detailed per-job match analysis
     - Skills gap report
     - Resume improvement suggestions
     - Overall candidacy summary

Groq is used because:
  - Free tier with generous limits
  - Fast inference (~200 tokens/sec)
  - No credit card needed for signup
  - llama3-8b is highly capable for structured analysis
"""

import os
import json
from typing import Optional

# Groq client — falls back gracefully if not installed/configured
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


GROQ_MODEL = "llama3-8b-8192"

ANALYSIS_PROMPT = """You are an expert career counselor and AI recruiter. 
You have been given a candidate's resume and a list of job matches retrieved from a vector database.

Your task is to perform a deep, structured analysis and return ONLY valid JSON — no markdown, no explanation outside the JSON.

Resume Summary:
{resume_text}

Matched Jobs (ranked by vector similarity):
{jobs_context}

Return this exact JSON structure:
{{
  "overall_summary": "2-3 sentence summary of the candidate's profile and overall fit",
  "candidacy_score": <integer 0-100 overall employability score>,
  "top_strength": "The single biggest strength this candidate has",
  "biggest_gap": "The single most important skill or experience gap",
  "job_analyses": [
    {{
      "job_id": "<id>",
      "fit_verdict": "Strong Fit" | "Good Fit" | "Partial Fit" | "Stretch Role",
      "why_good": "1-2 sentences on why this is a good match",
      "what_missing": "1-2 sentences on what's missing or needs strengthening",
      "interview_tip": "One specific tip for interviewing for this role"
    }}
  ],
  "skills_gap": [
    {{"skill": "<skill name>", "importance": "High"|"Medium"|"Low", "how_to_learn": "brief suggestion"}}
  ],
  "resume_improvements": [
    "Improvement suggestion 1",
    "Improvement suggestion 2",
    "Improvement suggestion 3"
  ],
  "next_steps": "Practical 2-3 sentence advice on what this candidate should do next"
}}"""


def build_jobs_context(jobs: list[dict]) -> str:
    """Serialize matched jobs into a compact context string for the LLM."""
    parts = []
    for i, job in enumerate(jobs, 1):
        skills_str = ", ".join(job.get("skills", [])[:8])
        parts.append(
            f"[{i}] ID:{job['id']} | {job['title']} at {job['company']} | "
            f"Exp: {job['experience']} | Salary: {job['salary']} | "
            f"Skills: {skills_str} | "
            f"Score: {round(job['similarity_score']*100,1)}%"
        )
    return "\n".join(parts)


def run_rag_analysis(
    resume_text: str,
    matched_jobs: list[dict],
    groq_api_key: Optional[str] = None,
) -> dict:
    """
    Core RAG function.
    Sends resume + retrieved job context to LLM → returns structured analysis.
    Falls back to rule-based analysis if Groq is not available.
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

    if not GROQ_AVAILABLE or not api_key:
        return _fallback_analysis(resume_text, matched_jobs)

    try:
        client = Groq(api_key=api_key)
        jobs_context = build_jobs_context(matched_jobs)

        prompt = ANALYSIS_PROMPT.format(
            resume_text=resume_text[:2000],  # cap to avoid token overflow
            jobs_context=jobs_context,
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        result["rag_source"] = "groq_llm"
        return result

    except json.JSONDecodeError:
        return _fallback_analysis(resume_text, matched_jobs)
    except Exception as e:
        print(f"Groq RAG error: {e}")
        return _fallback_analysis(resume_text, matched_jobs)


def _fallback_analysis(resume_text: str, matched_jobs: list[dict]) -> dict:
    """
    Rule-based fallback analysis when Groq is not available.
    Still produces useful structured output.
    """
    resume_lower = resume_text.lower()

    # Extract skills from resume text via keyword scan
    common_skills = [
        "python", "java", "javascript", "typescript", "react", "node",
        "sql", "docker", "kubernetes", "aws", "ml", "deep learning",
        "nlp", "pytorch", "tensorflow", "fastapi", "django", "flask",
        "spark", "kafka", "git", "linux", "rest api", "microservices",
        "data science", "machine learning", "llm", "transformers",
    ]
    found_skills = [s for s in common_skills if s in resume_lower]

    # Compute skills gap per job
    job_analyses = []
    all_job_skills = set()
    for job in matched_jobs:
        job_skills = set(s.lower() for s in job.get("skills", []))
        all_job_skills.update(job_skills)
        matched = set(found_skills) & job_skills
        score = job.get("similarity_score", 0)

        if score >= 0.75:
            verdict = "Strong Fit"
        elif score >= 0.60:
            verdict = "Good Fit"
        elif score >= 0.45:
            verdict = "Partial Fit"
        else:
            verdict = "Stretch Role"

        missing = job_skills - set(found_skills)
        missing_str = ", ".join(list(missing)[:3]) if missing else "No critical gaps detected"

        job_analyses.append({
            "job_id": job["id"],
            "fit_verdict": verdict,
            "why_good": f"Your profile shows {round(score*100,1)}% semantic alignment with this role. Shared skills: {', '.join(list(matched)[:3]) or 'conceptual overlap detected'}.",
            "what_missing": f"Consider strengthening: {missing_str}.",
            "interview_tip": f"Highlight your most relevant projects and quantify your impact when discussing {job['title']} responsibilities.",
        })

    # Skills gap
    resume_skill_set = set(found_skills)
    gap_skills = list(all_job_skills - resume_skill_set)[:5]
    skills_gap = []
    for skill in gap_skills:
        skills_gap.append({
            "skill": skill.title(),
            "importance": "High" if skill in ["python", "docker", "aws", "kubernetes", "react"] else "Medium",
            "how_to_learn": f"Take a hands-on course on {skill.title()} and build a small project",
        })

    overall_score = min(95, int(len(found_skills) * 6 + (matched_jobs[0]["similarity_score"] if matched_jobs else 0) * 30))

    return {
        "overall_summary": (
            f"This candidate has a solid technical foundation with {len(found_skills)} detectable skills. "
            f"The strongest match is for {matched_jobs[0]['title']} at {matched_jobs[0]['company']} "
            f"with {round(matched_jobs[0]['similarity_score']*100,1)}% semantic alignment."
            if matched_jobs else "Resume parsed successfully. Add a Groq API key for deeper AI analysis."
        ),
        "candidacy_score": overall_score,
        "top_strength": f"Technical breadth across {', '.join(found_skills[:3])}" if found_skills else "Diverse background",
        "biggest_gap": f"Could strengthen: {gap_skills[0].title() if gap_skills else 'No major gaps detected'}",
        "job_analyses": job_analyses,
        "skills_gap": skills_gap,
        "resume_improvements": [
            "Add quantified achievements (e.g. 'Improved model accuracy by 15%')",
            "Include links to GitHub profile and notable projects",
            "Tailor your summary section for each role you apply to",
        ],
        "next_steps": (
            "Focus on the top 2 matched roles and tailor your resume specifically for each. "
            "Fill identified skill gaps with project-based learning. "
            "Add a Groq API key to get deeper AI-powered analysis."
        ),
        "rag_source": "rule_based_fallback",
    }
