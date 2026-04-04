"""
streamlit_app.py - V2: Richer UI with score breakdown and skill gap analysis
"""

import streamlit as st
import requests

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Matcher V2",
    page_icon="📄",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📄 Resume Matcher — V2")
st.caption("Semantic similarity using Sentence Transformers (all-MiniLM-L6-v2)")

st.info(
    "**How scoring works:** 70% semantic similarity (embeddings) + "
    "30% skill match. Semantic similarity understands *meaning*, "
    "not just keyword overlap.",
    icon="ℹ️"
)

# ── Input Layout ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📎 Upload Resume (PDF)", type=["pdf"])

with col2:
    job_description = st.text_area(
        "📋 Paste Job Description",
        height=200,
        placeholder="Paste the full job description here..."
    )

analyze_btn = st.button("🔍 Analyze Match", type="primary", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze_btn:
    if uploaded_file is None:
        st.error("Please upload a resume PDF.")
        st.stop()

    with st.spinner("Running semantic analysis... (first run may take ~10s to load the model)"):
        files = {"resume": (uploaded_file.name, uploaded_file, "application/pdf")}
        data = {"job_description": job_description}

        try:
            response = requests.post(
                "http://127.0.0.1:5000/parse-resume",
                files=files,
                data=data,
                timeout=60
            )
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to Flask server. Make sure `app.py` is running.")
            st.code("python app.py", language="bash")
            st.stop()

    if response.status_code != 200:
        st.error(f"Server error: {response.text}")
        st.stop()

    result = response.json()

    # ── Candidate Info ────────────────────────────────────────────────────────
    st.success("✅ Analysis complete!")
    st.divider()

    st.subheader("👤 Candidate Info")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    info_col1.metric("Name", result.get("candidate_name") or "Not found")
    info_col2.metric("Email", result.get("email") or "Not found")
    info_col3.metric("Phone", result.get("phone") or "Not found")
    info_col4.metric(
        "Est. Experience",
        f"{result.get('estimated_years_experience')} yrs"
        if result.get("estimated_years_experience") else "Not found"
    )

    st.divider()

    # ── Match Scores ──────────────────────────────────────────────────────────
    if result.get("final_match_score") is not None:
        st.subheader("🎯 Match Scores")

        score_col1, score_col2, score_col3 = st.columns(3)

        final = result["final_match_score"]
        semantic = result["semantic_similarity"]
        skill = result["skill_match_score"]

        # Color-coded final score
        score_color = (
            "🟢" if final >= 0.8 else
            "🟡" if final >= 0.6 else
            "🟠" if final >= 0.4 else "🔴"
        )
        guide = result.get("_score_guide", {})
        label = next((v for k, v in guide.items()
                      if float(k.split(" - ")[0]) <= final <= float(k.split(" - ")[1])), "")

        score_col1.metric(
            f"{score_color} Final Match Score",
            f"{final:.1%}",
            label
        )
        score_col2.metric(
            "🧠 Semantic Similarity (70%)",
            f"{semantic:.1%}",
            "Embedding-based — understands meaning"
        )
        score_col3.metric(
            "🔑 Skill Match (30%)",
            f"{skill:.1%}",
            "Keyword matching against JD"
        )

        # Progress bar for final score
        st.progress(final, text=f"Overall match: {final:.1%}")

        st.divider()

    # ── Skills ────────────────────────────────────────────────────────────────
    st.subheader("🛠️ Skills Found in Resume")

    resume_skills = result.get("resume_skills", {})
    if resume_skills:
        skill_display = {k: v for k, v in resume_skills.items() if k != "all"}
        for category, skills in skill_display.items():
            if skills:
                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(skills)}")
    else:
        st.write("No skills detected from the taxonomy.")

    # ── Skill Gap Analysis ────────────────────────────────────────────────────
    missing = result.get("missing_skills", [])
    if missing:
        st.divider()
        st.subheader("⚠️ Skill Gap Analysis")
        st.warning(
            f"These skills were required in the JD but **not found** in the resume: "
            f"**{', '.join(missing)}**"
        )
        st.caption(
            "💡 Tip: Add these to your resume if you have experience with them, "
            "or prioritize learning them."
        )

    # ── Sections Detected ─────────────────────────────────────────────────────
    sections_found = result.get("sections_found", [])
    if sections_found:
        st.divider()
        st.subheader("📑 Resume Sections Detected")
        st.write(", ".join(s.title() for s in sections_found))

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("🔍 Raw API Response (for debugging)"):
        st.json(result)