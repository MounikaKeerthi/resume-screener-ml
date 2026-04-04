# Resume Screener — V2

NLP-powered resume matcher using semantic embeddings.

## How It Works
1. Upload a resume PDF
2. Paste a job description
3. Get a match score with skill gap analysis

## Tech Stack
- Sentence Transformers (all-MiniLM-L6-v2) — semantic similarity
- spaCy — NER, name extraction
- PyMuPDF — PDF parsing
- Flask — REST API
- Streamlit — UI

## Scoring
- 70% semantic similarity (SBERT embeddings)
- 30% skill match (taxonomy-based)

## Run Locally
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
streamlit run streamlit_app.py

## Roadmap
- V3: LLM-powered explanation via Claude API
- V4: Proper NER for skill extraction
- V5: Fine-tuned matching model
