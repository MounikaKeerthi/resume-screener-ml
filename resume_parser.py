import fitz  # PyMuPDF
import re

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

    return {
    "resume_email": extract_email(text),
    "resume_phone": extract_phone(text),
    "resume_skills": extract_skills(text),
    "job_description_provided": bool(job_description)
    }

