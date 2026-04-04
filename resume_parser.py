from typing import Optional
import fitz  # PyMuPDF
import re
import spacy
from nlp_utils import preprocess_for_embeddings, extract_sections, clean_pdf_text
from similarity import semantic_similarity

nlp = spacy.load("en_core_web_sm")

SKILL_TAXONOMY = {
    "languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "scala", "r", "kotlin", "swift", "php", "ruby"
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "reinforcement learning", "transformers", "llm",
        "neural network", "pytorch", "tensorflow", "keras", "scikit-learn",
        "hugging face", "langchain", "rag", "fine-tuning", "embeddings",
        "xgboost", "random forest"
    ],
    "data": [
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "pandas", "numpy", "spark", "hadoop", "dbt", "airflow",
        "tableau", "power bi", "data pipeline"
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
        "ci/cd", "github actions", "jenkins", "linux", "bash"
    ],
    "web_frameworks": [
        "flask", "fastapi", "django", "react", "node.js", "express",
        "rest api", "graphql", "microservices"
    ],
    "tools": [
        "git", "jira", "confluence", "postman", "jupyter", "vscode"
    ]
}

# Flatten for quick lookup, keeping a reverse map for category reporting
FLAT_SKILLS = {}
for category, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        FLAT_SKILLS[skill] = category

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract raw text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_email(text: str) -> Optional[str]:
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    phone_pattern = r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else None


def extract_name(text: str) -> Optional[str]:
    # Look at the header portion only
    header_text = clean_pdf_text(text[:300])
    doc = nlp(header_text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text

    return None


def extract_skills(text: str) -> dict:
    text_lower = text.lower()
    found_by_category = {cat: [] for cat in SKILL_TAXONOMY}
    all_found = []

    for skill, category in FLAT_SKILLS.items():
        # Use word boundaries to avoid "r" matching "docker"
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_by_category[category].append(skill)
            all_found.append(skill)

    # Remove empty categories for cleaner output
    found_by_category = {k: v for k, v in found_by_category.items() if v}
    found_by_category["all"] = all_found

    return found_by_category


def extract_years_of_experience(text: str) -> Optional[int]:
    # Pattern 1: Explicit mention ("5+ years of experience")
    explicit = re.search(
        r'(\d+)\+?\s*years?\s+of\s+(professional\s+)?(experience|expertise)',
        text.lower()
    )
    if explicit:
        return int(explicit.group(1))

    # Pattern 2: Count year ranges in work history (e.g., "2019 - 2023")
    year_ranges = re.findall(r'(20\d{2})\s*[-–]\s*(20\d{2}|present|current)', text.lower())
    if year_ranges:
        total_years = 0
        current_year = 2025
        for start, end in year_ranges:
            end_year = current_year if end in ("present", "current") else int(end)
            total_years += max(0, end_year - int(start))
        return total_years

    return None

def parse_resume(pdf_path: str, job_description: str = "") -> dict:
    # Step 1: Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)

    # Step 2: Parse structured fields
    name = extract_name(raw_text)
    email = extract_email(raw_text)
    phone = extract_phone(raw_text)
    resume_skills = extract_skills(raw_text)
    years_exp = extract_years_of_experience(raw_text)
    sections = extract_sections(raw_text)

    # Step 3: Prepare text for embedding (light cleaning, truncation)
    resume_text_clean = preprocess_for_embeddings(raw_text)

    # Step 4: Compute match scores if JD is provided
    semantic_score = None
    skill_match_score = None
    final_score = None
    jd_skills = {}
    missing_skills = []

    if job_description:
        jd_text_clean = preprocess_for_embeddings(job_description)

        # Semantic similarity: embeddings-based
        semantic_score = semantic_similarity(resume_text_clean, jd_text_clean)

        # Skill matching: which JD skills does the resume cover?
        jd_skills = extract_skills(job_description)
        jd_all_skills = set(jd_skills.get("all", []))
        resume_all_skills = set(resume_skills.get("all", []))

        if jd_all_skills:
            matched = resume_all_skills & jd_all_skills
            missing_skills = list(jd_all_skills - resume_all_skills)
            skill_match_score = len(matched) / len(jd_all_skills)
        else:
            skill_match_score = 0.0

        # Weighted final score
        final_score = round(0.7 * semantic_score + 0.3 * skill_match_score, 4)

    return {
        # Candidate info
        "candidate_name": name,
        "email": email,
        "phone": phone,
        "estimated_years_experience": years_exp,

        # Skills
        "resume_skills": resume_skills,
        "sections_found": list(sections.keys()),

        # Match scores (only present if JD was provided)
        "semantic_similarity": round(semantic_score, 4) if semantic_score is not None else None,
        "skill_match_score": round(skill_match_score, 4) if skill_match_score is not None else None,
        "final_match_score": final_score,

        # Actionable feedback
        "jd_skills_detected": jd_skills.get("all", []),
        "missing_skills": missing_skills,

        # Interpretation guide
        "_score_guide": {
            "0.8 - 1.0": "Excellent match",
            "0.6 - 0.8": "Good match",
            "0.4 - 0.6": "Partial match",
            "0.0 - 0.4": "Low match"
        }
    }