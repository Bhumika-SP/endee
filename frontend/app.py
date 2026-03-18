"""
app.py — Smart Resume–Job Matching System
Two modes:
  1. Profile Search  — manual input → semantic Endee search
  2. Resume Analyzer — upload PDF   → Endee retrieval + RAG deep analysis
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="TalentMatch AI — Powered by Endee",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #0f0e17;
}
.stApp { background: #f4f1eb; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 5rem !important; max-width: 1280px !important; }

/* ── HERO ── */
.hero-wrap {
    background: #0f0e17;
    border-radius: 24px;
    padding: 56px 60px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content:'';
    position:absolute; top:-120px; right:-80px;
    width:500px; height:500px;
    background: radial-gradient(circle, rgba(255,107,53,0.12) 0%, transparent 65%);
    border-radius:50%;
    pointer-events:none;
}
.hero-wrap::after {
    content:'';
    position:absolute; bottom:-100px; left:30%;
    width:600px; height:300px;
    background: radial-gradient(ellipse, rgba(100,149,237,0.09) 0%, transparent 65%);
    pointer-events:none;
}
.hero-tag {
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(255,107,53,0.15);
    border:1px solid rgba(255,107,53,0.3);
    border-radius:999px;
    padding:5px 14px;
    font-size:11px; font-weight:600; letter-spacing:2px;
    text-transform:uppercase; color:#ff6b35;
    margin-bottom:20px;
}
.hero-title {
    font-family:'Syne',sans-serif;
    font-size:56px; font-weight:800; color:#fff;
    line-height:1.05; margin:0 0 18px;
}
.hero-title em { color:#ff6b35; font-style:normal; }
.hero-sub {
    font-size:16px; font-weight:300;
    color:rgba(255,255,255,0.55); max-width:520px; line-height:1.7;
}
.hero-stats {
    display:flex; gap:32px; margin-top:36px; flex-wrap:wrap;
}
.hero-stat-num {
    font-family:'Syne',sans-serif;
    font-size:28px; font-weight:800; color:#fff;
}
.hero-stat-lbl {
    font-size:11px; color:rgba(255,255,255,0.4);
    text-transform:uppercase; letter-spacing:1.5px; margin-top:2px;
}

/* ── MODE TABS ── */
.mode-tabs {
    display:flex; gap:4px;
    background:#e8e4dc;
    border-radius:14px;
    padding:4px;
    margin-bottom:28px;
    width:fit-content;
}
.mode-tab {
    padding:10px 28px;
    border-radius:10px;
    font-family:'Syne',sans-serif;
    font-size:13px; font-weight:700;
    cursor:pointer;
    border:none;
    transition:all .2s;
    letter-spacing:.5px;
}
.mode-tab.active {
    background:#0f0e17; color:#fff;
    box-shadow:0 2px 8px rgba(0,0,0,0.15);
}
.mode-tab.inactive {
    background:transparent; color:#888;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-family:'Syne',sans-serif;
    font-size:10px; font-weight:700;
    letter-spacing:3px; text-transform:uppercase;
    color:#aaa; margin-bottom:14px;
}

/* ── CARDS (generic) ── */
.card {
    background:#fff;
    border-radius:18px;
    border:1.5px solid #e8e4dc;
    padding:28px 30px;
    margin-bottom:16px;
}

/* ── STREAMLIT INPUT OVERRIDES ── */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
    background:#f4f1eb !important;
    border:1.5px solid #ddd8ce !important;
    border-radius:10px !important;
    font-family:'Inter',sans-serif !important;
    font-size:14px !important; color:#0f0e17 !important;
    padding:12px 16px !important;
}
.stTextInput>div>div>input:focus,
.stTextArea>div>div>textarea:focus {
    border-color:#0f0e17 !important;
    box-shadow:0 0 0 3px rgba(15,14,23,0.08) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-family:'Inter',sans-serif !important;
    font-size:13px !important; font-weight:500 !important;
    color:#555 !important; margin-bottom:6px !important;
}
.stSelectbox>div>div {
    background:#f4f1eb !important;
    border:1.5px solid #ddd8ce !important;
    border-radius:10px !important;
}

/* ── BUTTONS ── */
.stButton>button {
    background:#0f0e17 !important; color:#fff !important;
    border:none !important; border-radius:10px !important;
    font-family:'Syne',sans-serif !important;
    font-size:13px !important; font-weight:700 !important;
    letter-spacing:1px !important;
    padding:14px 28px !important; width:100% !important;
    cursor:pointer !important;
}
.stButton>button:hover { background:#2a2940 !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background:#fff !important;
    border:2px dashed #ddd8ce !important;
    border-radius:14px !important;
    padding:8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color:#ff6b35 !important;
}

/* ── JOB CARD ── */
.job-card {
    background:#fff;
    border-radius:18px;
    border:1.5px solid #e8e4dc;
    padding:26px 28px;
    margin-bottom:14px;
    position:relative;
    transition:transform .15s, box-shadow .15s;
}
.job-card:hover {
    transform:translateY(-2px);
    box-shadow:0 10px 36px rgba(15,14,23,0.09);
}
.job-rank-badge {
    position:absolute; top:22px; right:22px;
    background:#0f0e17; color:#fff;
    border-radius:8px;
    font-family:'Syne',sans-serif;
    font-size:11px; font-weight:700;
    padding:3px 10px;
}
.jc-title {
    font-family:'Syne',sans-serif;
    font-size:19px; font-weight:700;
    color:#0f0e17; margin:0 0 3px;
}
.jc-company {
    font-size:13px; font-weight:600;
    color:#ff6b35; margin-bottom:14px;
}
.pill-row { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:14px; }
.pill {
    background:#f4f1eb; border:1px solid #e0dbd2;
    border-radius:999px; padding:4px 12px;
    font-size:11px; color:#666;
}
.pill.green {
    background:#edfaf3; border-color:#a8dfc0;
    color:#1a6b3c; font-weight:600;
}
.pill.orange {
    background:#fff4f0; border-color:#ffc5ab;
    color:#c0400a; font-weight:600;
}
.jc-desc {
    font-size:13px; color:#666;
    line-height:1.65; margin-bottom:14px;
}
.skill-tag {
    display:inline-block;
    background:#0f0e17; color:rgba(255,255,255,.8);
    border-radius:6px; padding:3px 9px;
    font-size:10px; font-weight:500;
    font-family:'JetBrains Mono',monospace;
    margin:2px;
}
.skill-tag.matched {
    background:#ff6b35; color:#fff;
}
.match-footer {
    border-top:1px solid #f0ece4;
    padding-top:14px; margin-top:8px;
}
.match-row {
    display:flex; justify-content:space-between;
    align-items:center; margin-bottom:8px;
}
.match-lbl {
    font-size:10px; font-weight:700;
    text-transform:uppercase; letter-spacing:1.5px; color:#bbb;
}
.match-pct {
    font-family:'Syne',sans-serif;
    font-size:20px; font-weight:800; color:#0f0e17;
}
.bar-bg {
    background:#f0ece4; border-radius:999px;
    height:5px; overflow:hidden; margin-bottom:6px;
}
.bar-fill { height:100%; border-radius:999px; }
.match-why {
    font-size:11px; color:#aaa;
    font-style:italic; margin-top:4px;
}

/* ── RAG ANALYSIS PANEL ── */
.rag-panel {
    background:#0f0e17;
    border-radius:20px;
    padding:32px 36px;
    margin-bottom:28px;
    color:#fff;
}
.rag-title {
    font-family:'Syne',sans-serif;
    font-size:22px; font-weight:800;
    color:#fff; margin-bottom:6px;
}
.rag-sub { font-size:13px; color:rgba(255,255,255,.45); margin-bottom:24px; }
.rag-score-wrap {
    display:flex; align-items:flex-end; gap:8px; margin-bottom:24px;
}
.rag-score-num {
    font-family:'Syne',sans-serif;
    font-size:64px; font-weight:800;
    color:#ff6b35; line-height:1;
}
.rag-score-lbl {
    font-size:12px; color:rgba(255,255,255,.4);
    text-transform:uppercase; letter-spacing:2px;
    margin-bottom:10px;
}
.rag-summary {
    font-size:14px; color:rgba(255,255,255,.7);
    line-height:1.75; margin-bottom:24px;
    border-left:3px solid #ff6b35;
    padding-left:16px;
}
.rag-grid {
    display:grid; grid-template-columns:1fr 1fr;
    gap:14px; margin-bottom:20px;
}
.rag-cell {
    background:rgba(255,255,255,.05);
    border:1px solid rgba(255,255,255,.08);
    border-radius:12px; padding:16px;
}
.rag-cell-lbl {
    font-size:9px; font-weight:700;
    letter-spacing:2px; text-transform:uppercase;
    color:rgba(255,255,255,.3); margin-bottom:6px;
}
.rag-cell-val {
    font-size:13px; color:rgba(255,255,255,.85); line-height:1.5;
}
.rag-section-head {
    font-family:'Syne',sans-serif;
    font-size:12px; font-weight:700;
    letter-spacing:2px; text-transform:uppercase;
    color:rgba(255,255,255,.35); margin:20px 0 10px;
}
.gap-item {
    display:flex; align-items:flex-start; gap:10px;
    background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.07);
    border-radius:10px; padding:12px 14px;
    margin-bottom:8px;
}
.gap-skill {
    font-family:'JetBrains Mono',monospace;
    font-size:12px; font-weight:500; color:#ff6b35;
    min-width:90px;
}
.gap-imp {
    font-size:10px; font-weight:700;
    text-transform:uppercase; letter-spacing:1px;
    padding:2px 8px; border-radius:4px; margin-right:8px;
}
.gap-imp.high { background:rgba(255,107,53,.2); color:#ff6b35; }
.gap-imp.medium { background:rgba(255,193,7,.15); color:#ffc107; }
.gap-imp.low { background:rgba(100,149,237,.15); color:#6495ed; }
.gap-how { font-size:12px; color:rgba(255,255,255,.5); }
.improve-item {
    display:flex; gap:10px; align-items:flex-start;
    margin-bottom:8px;
}
.improve-dot {
    width:6px; height:6px;
    background:#ff6b35; border-radius:50%;
    flex-shrink:0; margin-top:6px;
}
.improve-text { font-size:13px; color:rgba(255,255,255,.65); line-height:1.55; }
.next-steps-box {
    background:rgba(255,107,53,.1);
    border:1px solid rgba(255,107,53,.25);
    border-radius:12px; padding:16px 18px;
    font-size:13px; color:rgba(255,255,255,.75);
    line-height:1.7; margin-top:16px;
}
.fit-badge {
    display:inline-block;
    border-radius:6px; padding:3px 10px;
    font-size:10px; font-weight:700;
    text-transform:uppercase; letter-spacing:.5px;
}
.fit-strong { background:rgba(0,200,100,.15); color:#00c864; }
.fit-good   { background:rgba(100,149,237,.15); color:#6495ed; }
.fit-partial{ background:rgba(255,193,7,.15); color:#ffc107; }
.fit-stretch{ background:rgba(255,107,53,.15); color:#ff6b35; }

/* ── QUERY PILL ── */
.query-pill {
    background:#ede9e0; border-radius:10px;
    padding:11px 16px; font-size:12px; color:#888;
    margin-bottom:22px; line-height:1.5;
}
.query-pill strong { color:#0f0e17; }

/* ── ENDEE FOOTER BADGE ── */
.endee-footer {
    display:inline-flex; align-items:center; gap:8px;
    background:#ede9e0; border:1px solid #ddd8ce;
    border-radius:10px; padding:10px 16px;
    font-size:11px; color:#888; margin-top:36px;
}
.endee-footer strong { color:#0f0e17; }

/* ── RESUME INFO CARD ── */
.resume-info {
    background:#fff;
    border:1.5px solid #e8e4dc;
    border-radius:14px; padding:20px 24px;
    margin-bottom:20px;
}
.ri-row { display:flex; gap:32px; flex-wrap:wrap; }
.ri-item-lbl {
    font-size:9px; font-weight:700;
    letter-spacing:2px; text-transform:uppercase;
    color:#bbb; margin-bottom:4px;
}
.ri-item-val {
    font-size:14px; font-weight:600; color:#0f0e17;
}

/* ── HOW IT WORKS CARD ── */
.hiw-card {
    background:#fff;
    border:1.5px solid #e8e4dc;
    border-radius:16px; padding:22px 24px;
}
.hiw-title {
    font-family:'Syne',sans-serif;
    font-size:13px; font-weight:700; color:#0f0e17;
    margin-bottom:14px;
}
.hiw-step { display:flex; gap:12px; margin-bottom:10px; }
.hiw-num {
    width:22px; height:22px; flex-shrink:0;
    background:#0f0e17; color:#fff;
    border-radius:6px; font-family:'Syne',sans-serif;
    font-size:11px; font-weight:700;
    display:flex; align-items:center; justify-content:center;
}
.hiw-text { font-size:12px; color:#888; line-height:1.6; }
.hiw-text b { color:#0f0e17; }

/* ── SPINNER ── */
.stSpinner>div { border-top-color:#ff6b35 !important; }

/* ── GROQ INPUT ── */
.groq-box {
    background:#fff8f5;
    border:1.5px solid #ffd0bc;
    border-radius:12px; padding:16px 18px;
    margin-top:12px;
}
.groq-box-title {
    font-size:12px; font-weight:600; color:#c0400a;
    margin-bottom:6px;
}
.groq-box-sub { font-size:11px; color:#e07050; margin-bottom:8px; }

/* ── EMPTY STATE ── */
.empty-state {
    text-align:center; padding:70px 0 50px;
}
.empty-icon { font-size:52px; margin-bottom:14px; }
.empty-title {
    font-family:'Syne',sans-serif;
    font-size:20px; font-weight:700; color:#ccc;
}
.empty-sub { font-size:13px; color:#bbb; margin-top:8px; }

/* ── ALERT overrides ── */
.stAlert { border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">✦ AI-Powered Career Intelligence</div>
    <div class="hero-title">
        Match your <em>talent</em><br>to the right opportunity
    </div>
    <div class="hero-sub">
        Upload your resume or describe yourself — our AI uses <strong style="color:rgba(255,255,255,.75)">semantic vector search</strong>
        via Endee and <strong style="color:rgba(255,255,255,.75)">RAG analysis</strong> to find and explain your best job matches.
    </div>
    <div class="hero-stats">
        <div>
            <div class="hero-stat-num">20+</div>
            <div class="hero-stat-lbl">Jobs Indexed</div>
        </div>
        <div>
            <div class="hero-stat-num">384D</div>
            <div class="hero-stat-lbl">Vector Space</div>
        </div>
        <div>
            <div class="hero-stat-num">RAG</div>
            <div class="hero-stat-lbl">Deep Analysis</div>
        </div>
        <div>
            <div class="hero-stat-num">Endee</div>
            <div class="hero-stat-lbl">Vector DB</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODE SELECTOR (stored in session)
# ─────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "profile"

col_tab1, col_tab2, _ = st.columns([1.4, 1.6, 5])
with col_tab1:
    if st.button("🔍  Profile Search", key="tab_profile"):
        st.session_state.mode = "profile"
        st.rerun()
with col_tab2:
    if st.button("📄  Resume Analyzer", key="tab_resume"):
        st.session_state.mode = "resume"
        st.rerun()

# Visual active indicator
active_profile = "background:#0f0e17;color:#fff;border-radius:10px;padding:6px 18px;font-weight:700;" if st.session_state.mode == "profile" else "color:#999;padding:6px 18px;"
active_resume  = "background:#0f0e17;color:#fff;border-radius:10px;padding:6px 18px;font-weight:700;" if st.session_state.mode == "resume"  else "color:#999;padding:6px 18px;"

st.markdown(f"""
<div style="display:flex;gap:4px;background:#e8e4dc;border-radius:12px;
            padding:4px;width:fit-content;margin-bottom:28px;font-family:Syne,sans-serif;font-size:12px;letter-spacing:.5px;">
    <span style="{active_profile}">🔍 Profile Search</span>
    <span style="{active_resume}">📄 Resume Analyzer</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: render job cards
# ─────────────────────────────────────────────
def render_job_card(job: dict, rank: int, user_skills_str: str = ""):
    score   = job["similarity_score"]
    pct     = round(score * 100, 1)
    bar_w   = min(100, int(score * 100))

    if pct >= 72:
        bar_color = "linear-gradient(90deg,#00c864,#00a050)"
        pct_color = "#00a050"
    elif pct >= 55:
        bar_color = "linear-gradient(90deg,#6495ed,#4275cc)"
        pct_color = "#4275cc"
    else:
        bar_color = "linear-gradient(90deg,#ff6b35,#e04010)"
        pct_color = "#e04010"

    user_set = set(s.strip().lower() for s in user_skills_str.replace(",", " ").split())
    skills_html = ""
    for s in job["skills"][:9]:
        cls = "skill-tag matched" if s.lower() in user_set else "skill-tag"
        skills_html += f'<span class="{cls}">{s}</span>'

    salary_pill = f'<span class="pill green">💰 {job["salary"]}</span>'

    st.markdown(f"""
    <div class="job-card">
        <div class="job-rank-badge">#{rank}</div>
        <div class="jc-title">{job['title']}</div>
        <div class="jc-company">{job['company']}</div>
        <div class="pill-row">
            <span class="pill">📍 {job['location']}</span>
            <span class="pill">⏱ {job['experience']}</span>
            <span class="pill">💼 {job['type']}</span>
            {salary_pill}
        </div>
        <div class="jc-desc">{job['description'][:260]}{'…' if len(job['description'])>260 else ''}</div>
        <div style="margin-bottom:14px">{skills_html}</div>
        <div class="match-footer">
            <div class="match-row">
                <span class="match-lbl">Semantic Match</span>
                <span class="match-pct" style="color:{pct_color}">{pct}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{bar_w}%;background:{bar_color}"></div>
            </div>
            <div class="match-why">↳ {job['match_reason']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: render RAG analysis
# ─────────────────────────────────────────────
def render_rag_analysis(rag: dict, matches: list):
    score   = rag.get("candidacy_score", 0)
    summary = rag.get("overall_summary", "")
    source  = rag.get("rag_source", "")
    source_badge = (
        '<span style="background:rgba(255,107,53,.2);color:#ff6b35;'
        'border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;'
        'letter-spacing:1px;text-transform:uppercase;margin-left:8px;">✦ Groq LLM</span>'
        if source == "groq_llm" else
        '<span style="background:rgba(255,255,255,.1);color:rgba(255,255,255,.4);'
        'border-radius:6px;padding:2px 8px;font-size:10px;letter-spacing:1px;margin-left:8px;">Rule-based</span>'
    )

    # Build job analysis lookup
    job_analysis = {ja["job_id"]: ja for ja in rag.get("job_analyses", [])}

    fit_class_map = {
        "Strong Fit": "fit-strong",
        "Good Fit":   "fit-good",
        "Partial Fit":"fit-partial",
        "Stretch Role":"fit-stretch",
    }

    # Per-job verdicts inline (compact)
    job_verdicts_html = ""
    for m in matches[:5]:
        ja = job_analysis.get(m["id"], {})
        verdict   = ja.get("fit_verdict", "—")
        why_good  = ja.get("why_good", "")
        what_miss = ja.get("what_missing", "")
        tip       = ja.get("interview_tip", "")
        fc        = fit_class_map.get(verdict, "fit-partial")
        job_verdicts_html += f"""
        <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
                    border-radius:12px;padding:14px 16px;margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <span style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;
                             color:#fff;">{m['title']}</span>
                <span style="font-size:11px;color:rgba(255,255,255,.35);">@ {m['company']}</span>
                <span class="fit-badge {fc}">{verdict}</span>
            </div>
            {"<div style='font-size:12px;color:rgba(255,255,255,.6);margin-bottom:5px;'>✅ " + why_good + "</div>" if why_good else ""}
            {"<div style='font-size:12px;color:rgba(255,255,255,.45);margin-bottom:5px;'>⚠️ " + what_miss + "</div>" if what_miss else ""}
            {"<div style='font-size:11px;color:#ff6b35;'>💡 " + tip + "</div>" if tip else ""}
        </div>"""

    # Skills gap
    gaps = rag.get("skills_gap", [])
    gaps_html = ""
    for g in gaps[:5]:
        imp = g.get("importance","Medium").lower()
        gaps_html += f"""
        <div class="gap-item">
            <span class="gap-skill">{g.get('skill','')}</span>
            <span class="gap-imp {imp}">{g.get('importance','')}</span>
            <span class="gap-how">{g.get('how_to_learn','')}</span>
        </div>"""

    # Resume improvements
    improvements = rag.get("resume_improvements", [])
    impr_html = "".join(
        f'<div class="improve-item"><div class="improve-dot"></div>'
        f'<div class="improve-text">{tip}</div></div>'
        for tip in improvements
    )

    next_steps = rag.get("next_steps", "")

    st.markdown(f"""
    <div class="rag-panel">
        <div class="rag-title">AI Analysis Report {source_badge}</div>
        <div class="rag-sub">Generated using Endee retrieval + LLM reasoning on your resume</div>

        <div class="rag-score-wrap">
            <div class="rag-score-num">{score}</div>
            <div class="rag-score-lbl">/ 100<br>Candidacy<br>Score</div>
        </div>

        <div class="rag-summary">{summary}</div>

        <div class="rag-grid">
            <div class="rag-cell">
                <div class="rag-cell-lbl">Top Strength</div>
                <div class="rag-cell-val">{rag.get('top_strength','—')}</div>
            </div>
            <div class="rag-cell">
                <div class="rag-cell-lbl">Key Gap</div>
                <div class="rag-cell-val">{rag.get('biggest_gap','—')}</div>
            </div>
        </div>

        <div class="rag-section-head">Per-Job Fit Analysis</div>
        {job_verdicts_html}

        {"<div class='rag-section-head'>Skills Gap Report</div>" + gaps_html if gaps_html else ""}

        {"<div class='rag-section-head'>Resume Improvements</div>" + impr_html if impr_html else ""}

        {"<div class='rag-section-head'>Next Steps</div><div class='next-steps-box'>" + next_steps + "</div>" if next_steps else ""}
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════
# MODE 1 — PROFILE SEARCH
# ═════════════════════════════════════════════
if st.session_state.mode == "profile":

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="sec-label">Your Profile</div>', unsafe_allow_html=True)
        name         = st.text_input("Your name", placeholder="e.g. Arjun Sharma")
        desired_role = st.text_input("What role are you targeting?",
                                     placeholder="e.g. Machine Learning Engineer focused on NLP")
        skills       = st.text_area("Your skills (comma-separated)",
                                    placeholder="e.g. Python, PyTorch, NLP, Transformers, Docker",
                                    height=90)
        experience   = st.text_area("Brief experience summary",
                                    placeholder="e.g. 2 years building text classification models at a fintech startup",
                                    height=90)

    with right:
        st.markdown('<div class="sec-label">Settings</div>', unsafe_allow_html=True)
        top_k = st.selectbox("Results to return", [3, 5, 8, 10], index=1)

        st.markdown("""
        <div class="hiw-card" style="margin-top:16px;">
            <div class="hiw-title">How Profile Search works</div>
            <div class="hiw-step"><div class="hiw-num">1</div>
                <div class="hiw-text">Profile → <b>sentence-transformer</b> → 384-dim vector</div></div>
            <div class="hiw-step"><div class="hiw-num">2</div>
                <div class="hiw-text"><b>Endee</b> cosine search across all job vectors</div></div>
            <div class="hiw-step"><div class="hiw-num">3</div>
                <div class="hiw-text">Results ranked by <b>semantic similarity</b> score</div></div>
            <div class="hiw-step"><div class="hiw-num">4</div>
                <div class="hiw-text">Matched skills <b style="color:#ff6b35">highlighted in orange</b></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔍  Find My Matches", key="btn_profile_search")

    if search_btn:
        if not desired_role or not skills:
            st.error("Please fill in at least your desired role and skills.")
        else:
            with st.spinner("Querying Endee vector database..."):
                try:
                    resp = requests.post(f"{API_URL}/recommend", json={
                        "name": name or "Candidate",
                        "desired_role": desired_role,
                        "skills": skills,
                        "experience": experience or "Not specified",
                        "top_k": top_k,
                    }, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach API. Run: `uvicorn backend.main:app --reload --port 8000`")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.stop()

            matches = data.get("matches", [])
            if not matches:
                st.warning("No matches. Try different keywords.")
                st.stop()

            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;margin:36px 0 18px;">
                <div style="font-family:Syne,sans-serif;font-size:24px;font-weight:800;color:#0f0e17;">
                    Top matches for <span style="color:#ff6b35">{data['candidate_name']}</span>
                </div>
                <div style="background:#0f0e17;color:#fff;border-radius:999px;padding:5px 16px;
                            font-size:12px;font-weight:600;">{len(matches)} results</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="query-pill">
                <strong>Query sent to Endee:</strong> "{data['query_used']}"
            </div>""", unsafe_allow_html=True)

            for rank, job in enumerate(matches, 1):
                render_job_card(job, rank, skills)

            st.markdown(f"""
            <div class="endee-footer">
                🗄️ <strong>{data['total_indexed']} vectors</strong> in Endee
                &nbsp;|&nbsp; Model: <strong>all-MiniLM-L6-v2</strong> (384-dim cosine)
                &nbsp;|&nbsp; Index: <strong>job_listings</strong>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">Enter your profile and search</div>
            <div class="empty-sub">Semantic results will appear here</div>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# MODE 2 — RESUME ANALYZER (RAG)
# ═════════════════════════════════════════════
else:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="sec-label">Upload Your Resume</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop your resume PDF here",
            type=["pdf"],
            help="Text-based PDF only (not scanned images). Max 5MB.",
        )
        top_k = st.selectbox("Number of job matches", [3, 5, 8, 10], index=1)

    with right:
        st.markdown('<div class="sec-label">AI Analysis Settings</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="groq-box">
            <div class="groq-box-title">🤖 Groq API Key (optional but recommended)</div>
            <div class="groq-box-sub">
                Add a free Groq key for deep LLM analysis. Without it, rule-based analysis is used.<br>
                Get yours free at <strong>console.groq.com</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        groq_key = st.text_input(
            "Groq API key",
            type="password",
            placeholder="gsk_...",
            help="Free at console.groq.com — no credit card needed",
        )

        st.markdown("""
        <div class="hiw-card" style="margin-top:16px;">
            <div class="hiw-title">How Resume Analyzer works</div>
            <div class="hiw-step"><div class="hiw-num">1</div>
                <div class="hiw-text">PDF parsed → skills, experience, education <b>extracted</b></div></div>
            <div class="hiw-step"><div class="hiw-num">2</div>
                <div class="hiw-text">Resume embedded → <b>Endee</b> retrieves top matches</div></div>
            <div class="hiw-step"><div class="hiw-num">3</div>
                <div class="hiw-text"><b>RAG:</b> resume + jobs → LLM deep analysis</div></div>
            <div class="hiw-step"><div class="hiw-num">4</div>
                <div class="hiw-text">Get <b>fit verdict</b>, skills gap, and improvement tips</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("✦  Analyze My Resume", key="btn_resume_analyze")

    if analyze_btn:
        if not uploaded:
            st.error("Please upload a PDF resume first.")
        else:
            with st.spinner("Parsing resume → querying Endee → running RAG analysis..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/resume/analyze",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        data={"top_k": top_k, "groq_api_key": groq_key or ""},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot reach API. Run: `uvicorn backend.main:app --reload --port 8000`")
                    st.stop()
                except requests.exceptions.HTTPError as e:
                    detail = resp.json().get("detail", str(e))
                    st.error(f"❌ {detail}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ {e}")
                    st.stop()

            # Resume info strip
            st.markdown(f"""
            <div class="resume-info">
                <div class="ri-row">
                    <div>
                        <div class="ri-item-lbl">Candidate</div>
                        <div class="ri-item-val">{data.get('candidate_name','—')}</div>
                    </div>
                    <div>
                        <div class="ri-item-lbl">Email</div>
                        <div class="ri-item-val">{data.get('email') or '—'}</div>
                    </div>
                    <div>
                        <div class="ri-item-lbl">Word count</div>
                        <div class="ri-item-val">{data.get('word_count','—')}</div>
                    </div>
                    <div>
                        <div class="ri-item-lbl">Jobs matched</div>
                        <div class="ri-item-val">{len(data.get('matches',[]))}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # RAG analysis panel (full dark panel)
            render_rag_analysis(data["rag_analysis"], data["matches"])

            # Job match cards
            st.markdown(f"""
            <div style="font-family:Syne,sans-serif;font-size:20px;font-weight:800;
                        color:#0f0e17;margin:28px 0 16px;">
                Top Job Matches <span style="color:#ff6b35">from Endee</span>
            </div>""", unsafe_allow_html=True)

            for rank, job in enumerate(data["matches"], 1):
                render_job_card(job, rank, "")

            st.markdown(f"""
            <div class="endee-footer">
                🗄️ <strong>{data['total_indexed']} vectors</strong> in Endee
                &nbsp;|&nbsp; Model: <strong>all-MiniLM-L6-v2</strong> (384-dim cosine)
                &nbsp;|&nbsp; RAG: <strong>{'Groq LLaMA3' if data['rag_analysis'].get('rag_source')=='groq_llm' else 'Rule-based'}</strong>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📄</div>
            <div class="empty-title">Upload your resume to begin</div>
            <div class="empty-sub">AI will analyze it and find your best matches</div>
        </div>""", unsafe_allow_html=True)
