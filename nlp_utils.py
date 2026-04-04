"""
nlp_utils.py - V2: Improved NLP Utilities

CONCEPT: Why preprocessing matters less for Sentence Transformers
-----------------------------------------------------------------
In V1 (TF-IDF), we aggressively preprocessed: lowercased, removed stopwords,
lemmatized. This was necessary because TF-IDF is sensitive to surface form —
"running" and "run" were treated as different features.

With Sentence Transformers, the model already handles this internally.
BERT-based models were trained on raw text, so:
  - Stopwords carry context (e.g., "not" matters!)
  - Capitalization can signal proper nouns
  - We should be LESS aggressive with cleaning

We still preprocess for two reasons:
  1. Remove PDF artifacts (weird characters, headers/footers)
  2. Keep preprocessing for the skill keyword matching pipeline
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")


def clean_pdf_text(text: str) -> str:
    """
    Light cleaning specifically for PDF-extracted text.

    PDFs often have:
      - Extra whitespace and newlines from layout
      - Page numbers and headers/footers
      - Hyphenated words broken across lines (e.g., "soft-\nware")
    """
    # Fix hyphenated line breaks: "soft-\nware" → "software"
    text = re.sub(r'-\n', '', text)

    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\s+', ' ', text)

    # Remove non-printable characters (PDF artifacts)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    return text.strip()


def preprocess_for_tfidf(text: str) -> str:
    """
    Aggressive preprocessing for TF-IDF / keyword-based matching.

    Used in skill extraction where we want exact keyword matching.
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)


def preprocess_for_embeddings(text: str) -> str:
    """
    Light preprocessing for Sentence Transformer input.

    We preserve sentence structure because the model was trained on
    natural sentences — mangling it hurts performance.
    """
    text = clean_pdf_text(text)

    # Truncate to ~512 words. BERT-based models have a 512 token limit.
    # Feeding more doesn't help — the model truncates internally anyway.
    # Better to be explicit and control what gets kept (the beginning
    # of a resume usually has the most important info).
    words = text.split()
    if len(words) > 512:
        text = " ".join(words[:512])

    return text


def extract_sections(text: str) -> dict:
    """
    Attempt to split a resume into logical sections.

    CONCEPT: Rule-based Information Extraction
    -------------------------------------------
    This is a classic NLP task: given unstructured text, find structure.
    We use regex patterns to detect section headers, then capture
    the text until the next header.

    This is V2's approach (regex). V3 will use ML-based section detection.
    """
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