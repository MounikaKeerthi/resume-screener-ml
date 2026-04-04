import streamlit as st
import tempfile
import os
from resume_parser import parse_resume
from similarity import semantic_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  /* Hero banner */
  .hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
  }
  .hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 0.5rem 0; }
  .hero p  { font-size: 1.05rem; opacity: 0.85; margin: 0; }

  /* Cards */
  .card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .card h3 { color: #1e40af; font-size: 1.05rem; margin: 0 0 1rem 0; }

  /* Score circle */
  .score-circle {
    width: 120px; height: 120px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin: 0 auto 1rem auto;
    font-weight: 800;
  }
  .score-number { font-size: 2rem; line-height: 1; }
  .score-label  { font-size: 0.7rem; opacity: 0.85; margin-top: 2px; }

  .score-high   { background: linear-gradient(135deg,#16a34a,#4ade80); color:white; }
  .score-medium { background: linear-gradient(135deg,#d97706,#fbbf24); color:white; }
  .score-low    { background: linear-gradient(135deg,#dc2626,#f87171); color:white; }

  /* Sub-scores */
  .sub-score-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0; border-bottom: 1px solid #f1f5f9;
  }
  .sub-score-row:last-child { border-bottom: none; }
  .sub-label { color: #475569; font-size: 0.9rem; }
  .sub-value { font-weight: 700; font-size: 0.95rem; }

  /* Skill chips */
  .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.5rem; }
  .chip {
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
  }
  .chip-found   { background:#dbeafe; color:#1d4ed8; }
  .chip-missing { background:#fee2e2; color:#b91c1c; }

  /* Analyze button */
  div.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
  }
  div.stButton > button:hover { opacity: 0.9 !important; }

  /* Info row in candidate panel */
  .info-row {
    display: flex; gap: 8px; align-items: flex-start;
    padding: 6px 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.9rem;
  }
  .info-row:last-child { border-bottom: none; }
  .info-key { color: #64748b; min-width: 120px; font-weight: 600; }
  .info-val { color: #1e293b; }

  /* Verdict banner */
  .verdict {
    border-radius: 10px; padding: 1rem 1.5rem;
    font-weight: 700; font-size: 1rem;
    margin-bottom: 1rem; text-align: center;
  }
  .verdict-strong { background:#dcfce7; color:#15803d; border:1px solid #86efac; }
  .verdict-good   { background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; }
  .verdict-low    { background:#fee2e2; color:#b91c1c; border:1px solid #fca5a5; }
</style>
""", unsafe_allow_html=True)

def score_color_class(score):
    if score >= 65: return "score-high"
    if score >= 40: return "score-medium"
    return "score-low"

def verdict_class(score):
    if score >= 65: return "verdict-strong", "🟢 Strong Match — Great fit for this role!"
    if score >= 40: return "verdict-good",   "🔵 Moderate Match — Some gaps to address."
    return "verdict-low", "🔴 Low Match — Significant skill gaps detected."


#Header
st.markdown("""
<div class="hero">
  <h1>📄 AI Resume Screener</h1>
  <p>Instantly match resumes against job descriptions using semantic AI analysis</p>
</div>
""", unsafe_allow_html=True)


#Inputs
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### 📤 Upload Resume (PDF)")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

with col_right:
    st.markdown("#### 📋 Paste Job Description")
    job_description = st.text_area(
        "", height=220,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed"
    )

st.markdown("---")
analyze_clicked = st.button("⚡ Analyze Resume", use_container_width=True)


#Analysis
if analyze_clicked:
    if not uploaded_file:
        st.warning("⚠️ Please upload a PDF resume.")
    elif not job_description.strip():
        st.warning("⚠️ Please paste a job description.")
    else:
        with st.spinner("Analyzing resume…"):
            try:
                # Save PDF to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Parse & score
                resume_data = parse_resume(tmp_path, job_description)
                results = resume_data
                os.unlink(tmp_path)

                final_score  = round(results.get("final_match_score", 0) * 100, 1)
                semantic     = round(results.get("semantic_similarity", 0) * 100, 1)
                skill_match  = round(results.get("skill_match_score", 0) * 100, 1)
                found_skills = found_skills = resume_data.get("resume_skills", {}).get("all", [])
                missing      = results.get("missing_skills", [])
                name         = resume_data.get("candidate_name", "—")
                exp          = resume_data.get("estimated_years_experience", "—")
                email        = resume_data.get("email", "—")

                #Verdict banner
                v_class, v_text = verdict_class(final_score)
                st.markdown(f'<div class="verdict {v_class}">{v_text}</div>', unsafe_allow_html=True)

                #Three columns: score | candidate | sub-scores
                c1, c2, c3 = st.columns([1, 1.4, 1.4], gap="large")

                # Overall score circle
                with c1:
                    sc = score_color_class(final_score)
                    st.markdown(f"""
                    <div class="card" style="text-align:center;">
                      <h3>Overall Match</h3>
                      <div class="score-circle {sc}">
                        <div class="score-number">{final_score}%</div>
                        <div class="score-label">MATCH SCORE</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                # Candidate info
                with c2:
                    st.markdown(f"""
                    <div class="card">
                      <h3>👤 Candidate Info</h3>
                      <div class="info-row"><span class="info-key">Name</span><span class="info-val">{name}</span></div>
                      <div class="info-row"><span class="info-key">Email</span><span class="info-val">{email}</span></div>
                      <div class="info-row"><span class="info-key">Experience</span><span class="info-val">{exp} years</span></div>
                      <div class="info-row"><span class="info-key">Skills Found</span><span class="info-val">{len(found_skills)}</span></div>
                      <div class="info-row"><span class="info-key">Skills Missing</span><span class="info-val">{len(missing)}</span></div>
                    </div>""", unsafe_allow_html=True)

                # Sub-scores
                with c3:
                    def sub_color(v):
                        if v >= 65: return "#16a34a"
                        if v >= 40: return "#d97706"
                        return "#dc2626"

                    st.markdown(f"""
                    <div class="card">
                      <h3>📊 Score Breakdown</h3>
                      <div class="sub-score-row">
                        <span class="sub-label">🧠 Semantic Match</span>
                        <span class="sub-value" style="color:{sub_color(semantic)}">{semantic}%</span>
                      </div>
                      <div class="sub-score-row">
                        <span class="sub-label">🛠️ Skill Match</span>
                        <span class="sub-value" style="color:{sub_color(skill_match)}">{skill_match}%</span>
                      </div>
                      <div class="sub-score-row">
                        <span class="sub-label">⭐ Final Score</span>
                        <span class="sub-value" style="color:{sub_color(final_score)}">{final_score}%</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                #Skills panels
                sk1, sk2 = st.columns(2, gap="large")

                with sk1:
                    chips = "".join(f'<span class="chip chip-found">{s}</span>' for s in found_skills[:20])
                    st.markdown(f"""
                    <div class="card">
                      <h3>✅ Skills Detected ({len(found_skills)})</h3>
                      <div class="chip-row">{chips if chips else "No skills detected"}</div>
                    </div>""", unsafe_allow_html=True)

                with sk2:
                    chips_m = "".join(f'<span class="chip chip-missing">{s}</span>' for s in missing[:20])
                    st.markdown(f"""
                    <div class="card">
                      <h3>❌ Missing Skills ({len(missing)})</h3>
                      <div class="chip-row">{chips_m if chips_m else "🎉 No major gaps found!"}</div>
                    </div>""", unsafe_allow_html=True)

                #Progress bars
                st.markdown("#### 📈 Visual Score Breakdown")
                st.progress(semantic / 100, text=f"Semantic Similarity: {semantic}%")
                st.progress(skill_match / 100, text=f"Skill Match: {skill_match}%")
                st.progress(final_score / 100, text=f"Final Score: {final_score}%")

            except Exception as e:
                st.error(f"Error during analysis: {e}")