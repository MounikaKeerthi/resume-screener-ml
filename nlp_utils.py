import re
import spacy

nlp = spacy.load("en_core_web_sm")


def clean_pdf_text(text: str) -> str:
    # Fix hyphenated line breaks: "soft-\nware" → "software"
    text = re.sub(r'-\n', '', text)

    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-printable characters (PDF artifacts)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    return text.strip()


def preprocess_for_tfidf(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)


def preprocess_for_embeddings(text: str) -> str:
    text = clean_pdf_text(text)
    words = text.split()
    if len(words) > 512:
        text = " ".join(words[:512])

    return text


def extract_sections(text: str) -> dict:
    text_lower = text.lower()

    # Common section header patterns in resumes
    section_patterns = {
        "summary":    r"(summary|objective|profile|about me)",
        "experience": r"(experience|work history|employment|work experience)",
        "education":  r"(education|academic|qualifications|degrees?)",
        "skills":     r"(skills|technical skills|competencies|technologies)",
        "projects":   r"(projects|personal projects|portfolio)",
        "certifications": r"(certifications?|certificates?|licenses?)",
    }

    # Find where each section starts
    section_starts = {}
    for section, pattern in section_patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            section_starts[section] = match.start()

    # Sort sections by their position in the document
    sorted_sections = sorted(section_starts.items(), key=lambda x: x[1])

    # Extract text between consecutive section starts
    sections = {}
    for i, (section_name, start_pos) in enumerate(sorted_sections):
        end_pos = sorted_sections[i + 1][1] if i + 1 < len(sorted_sections) else len(text)
        sections[section_name] = text[start_pos:end_pos].strip()

    return sections