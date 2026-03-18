"""
resume_parser.py — Extracts structured information from uploaded resume PDFs.

Supports:
  - PDF text extraction via PyMuPDF
  - Fallback plain text input
  - Section detection (Skills, Experience, Education, Projects)
"""

import re
import io
from typing import Optional


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")


def clean_text(text: str) -> str:
    """Clean extracted text — remove excessive whitespace and special chars."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def extract_email(text: str) -> Optional[str]:
    match = re.search(r'[\w.\-+]+@[\w\-]+\.[a-zA-Z]{2,}', text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    match = re.search(r'(\+?\d[\d\s\-().]{8,14}\d)', text)
    return match.group(0).strip() if match else None


def extract_name_heuristic(text: str) -> Optional[str]:
    """Try to extract candidate name from the top of the resume."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:5]:
        # Name is usually 2-4 words, no special chars, not all caps keyword
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            if not any(kw in line.upper() for kw in ['RESUME', 'CV', 'CURRICULUM', 'PROFILE', 'SUMMARY']):
                return line
    return None


SECTION_KEYWORDS = {
    "skills": ["skill", "technical skill", "core competenc", "technology", "tools", "stack"],
    "experience": ["experience", "work history", "employment", "career", "professional background", "internship"],
    "education": ["education", "academic", "qualification", "degree", "university", "college"],
    "projects": ["project", "portfolio", "work sample", "open source"],
    "summary": ["summary", "objective", "profile", "about me", "overview"],
    "certifications": ["certif", "award", "achiev", "honour", "honor"],
}


def detect_sections(text: str) -> dict:
    """
    Split resume text into named sections using heading detection.
    Returns dict of section_name -> content.
    """
    lines = text.split('\n')
    sections = {}
    current_section = "header"
    current_content = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_content.append('')
            continue

        # Detect section heading: short line that matches a keyword
        lower = stripped.lower()
        matched_section = None
        if len(stripped) < 60:  # headings are short
            for sec, keywords in SECTION_KEYWORDS.items():
                if any(kw in lower for kw in keywords):
                    matched_section = sec
                    break

        if matched_section:
            # Save old section
            sections[current_section] = '\n'.join(current_content).strip()
            current_section = matched_section
            current_content = []
        else:
            current_content.append(stripped)

    # Save last section
    sections[current_section] = '\n'.join(current_content).strip()
    return sections


def parse_resume(pdf_bytes: bytes) -> dict:
    """
    Full resume parsing pipeline.
    Returns structured dict with all extracted fields.
    """
    raw_text = extract_text_from_pdf(pdf_bytes)
    text = clean_text(raw_text)
    sections = detect_sections(text)

    return {
        "raw_text": text,
        "name": extract_name_heuristic(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "sections": sections,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


def build_resume_embedding_text(parsed: dict) -> str:
    """
    Build a rich text representation of the resume for embedding.
    Prioritises skills and experience sections.
    """
    parts = []
    sections = parsed.get("sections", {})

    if sections.get("summary"):
        parts.append(f"Professional Summary: {sections['summary']}")

    if sections.get("skills"):
        parts.append(f"Technical Skills: {sections['skills']}")

    if sections.get("experience"):
        exp = sections["experience"][:800]  # cap length
        parts.append(f"Work Experience: {exp}")

    if sections.get("projects"):
        proj = sections["projects"][:400]
        parts.append(f"Projects: {proj}")

    if sections.get("education"):
        parts.append(f"Education: {sections['education'][:200]}")

    if not parts:
        # Fallback: use raw text trimmed
        parts.append(parsed.get("raw_text", "")[:1200])

    return " | ".join(parts)
