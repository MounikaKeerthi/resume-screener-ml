import fitz  # PyMuPDF
import re
from nlp_utils import preprocess
from similarity import semantic_similarity

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_email(text):
    match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return match.group(0) if match else None


def extract_phone(text):
    match = re.search(r"\+?\d[\d -]{8,}\d", text)
    return match.group(0) if match else None


def extract_skills(text):
    skill_keywords = [
        "python", "java", "sql", "machine learning",
        "deep learning", "nlp", "flask", "react",
        "docker", "aws"
    ]

    text_lower = text.lower()
    found_skills = [skill for skill in skill_keywords if skill in text_lower]

    return found_skills


def parse_resume(pdf_path, job_description=""):
    text = extract_text_from_pdf(pdf_path)

    cleaned_resume = preprocess(text)
    cleaned_jd = preprocess(job_description) if job_description else ""

    semantic_score = (
        semantic_similarity(cleaned_resume, cleaned_jd)
        if job_description else None
    )

    resume_skills = extract_skills(text)
    jd_skills = extract_skills(job_description)

    skill_overlap = (
        len(set(resume_skills) & set(jd_skills)) / max(len(jd_skills), 1)
        if job_description else None
    )

    final_score = (
        0.7 * semantic_score + 0.3 * skill_overlap
        if semantic_score is not None else None
    )

    return {
        "resume_email": extract_email(text),
        "resume_phone": extract_phone(text),
        "resume_skills": resume_skills,
        "semantic_similarity": semantic_score,
        "skill_match_score": skill_overlap,
        "final_match_score": final_score
    }
