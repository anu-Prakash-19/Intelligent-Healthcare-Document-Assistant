# Intelligent Healthcare Document Assistant

End-to-end AI system for processing medical documents — OCR extraction, document classification, biomedical NER, risk detection, and drug-interaction decision support, behind a FastAPI + Streamlit app. Includes Qdrant-based multi-doc RAG search, automated PDF reports, and Docker deployment. Capstone project on MTSamples (~5,000 de-identified reports).

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?logo=qdrant&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)

---

## Overview

**Intelligent Healthcare Document Assistant** is an end-to-end AI pipeline for processing medical documents, extracting clinical insights, and supporting decision-making. It was built as a capstone project spanning research/experimentation (Google Colab) and a deployed application (FastAPI + Streamlit).

The pipeline ingests scanned or digital medical documents (PDF, DOCX, images), extracts structured clinical entities, classifies document type, flags risk indicators, checks for drug interactions, and generates polished PDF reports — all backed by a persistent, searchable multi-document library.

Dataset: [MTSamples](https://github.com/socd06/medical-nlp) — roughly 5,000 de-identified medical transcription reports (CC0 public domain).

---

## Key Features

- **Dual-path OCR extraction** — a unified `extract_text_from_file()` router dispatches by file type (PDF, DOCX, image)
- **Rule-based document classification** — specialty-aware keyword classifier tuned and evaluated on MTSamples
- **Biomedical NER** — parallel `d4data/biomedical-ner-all` + spaCy (`en_core_web_sm`) pipeline, with post-processing heuristics: word-boundary expansion, NegEx-style negation detection, and lab-value sanity checks
- **Risk detection engine** (`risk_engine.py`) — flags clinically significant findings from structured report data
- **Drug interaction & decision support** (`decision_support.py`) — frozenset-keyed interaction database for referral and safety checks, built as an extension layer over risk detection rather than a modification of it
- **Multi-document RAG library** — persistent Qdrant vector store supporting retrieval-augmented Q&A across documents, with `delete_document()` for right-to-erasure compliance
- **Automated report generation** — clinical PDF reports via ReportLab Platypus
- **Annotation system** (`annotation_service.py`) — sticky-note style commentary with document-specific targets generated from pipeline findings
- **Fine-tuning experiment** — Bio_ClinicalBERT fine-tuned on BC5CDR (F1 ≈ 0.84); documented as a validated methodological experiment, not deployed to production (see [Limitations](#limitations--compliance-notes))
- **Containerized deployment** — Docker-ready FastAPI backend + Streamlit frontend

---

## Architecture

The system is organized into 13 phases, spanning text extraction through report generation:

| Phase | Component | Description |
|---|---|---|
| 1–2 | Text Extraction | Dual-path OCR/document parsing router |
| 3 | Classification | Rule/keyword-based document type classifier |
| 4–5 | NER | Biomedical entity recognition + post-processing |
| 6 | Risk Detection | `risk_engine.py` |
| 10 | Document Library | Persistent Qdrant vector store, multi-document RAG |
| 11 | Fine-Tuning (Experimental) | Bio_ClinicalBERT on BC5CDR |
| 12 | Decision Support | `decision_support.py` — drug interactions |
| 13 | Report Generation | ReportLab Platypus PDF reports |

**Frontend:** Streamlit (`app.py`) with Plotly visualizations, multi-file upload support
**Backend:** FastAPI (`api.py`) served via `uvicorn`

---

## Tech Stack

- **Languages/Frameworks:** Python, Streamlit, FastAPI, uvicorn
- **ML/NLP:** HuggingFace Transformers (`d4data/biomedical-ner-all`, Bio_ClinicalBERT, distilbart-cnn, roberta-squad2, Helsinki-NLP opus-mt), spaCy, sentence-transformers
- **Vector Store:** Qdrant (production), ChromaDB (early experimentation)
- **LLM API:** Groq
- **Report Generation:** ReportLab Platypus
- **Containerization:** Docker
- **Dev Environment:** VS Code (Windows) for deployment, Google Colab for experimentation

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- A Groq API key

### Installation

```bash
git clone https://github.com/<your-username>/intelligent-healthcare-document-assistant.git
cd intelligent-healthcare-document-assistant
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with:

```
GROQ_API_KEY=your_key_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

### Running Locally

Start the backend:

```bash
uvicorn api:app --reload
```

Start the frontend:

```bash
streamlit run app.py
```

### Running with Docker

```bash
docker build -t healthcare-doc-assistant .
docker run -p 8000:8000 -p 8501:8501 healthcare-doc-assistant
```

---

## Limitations & Compliance Notes

This project is a capstone/research build, not a certified clinical or production healthcare system. A few things worth being upfront about:

- **No BAA for LLM API calls.** Q&A and translation features route through the Groq API, and no Business Associate Agreement is in place. This is the most significant compliance gap in the current architecture and is documented here rather than glossed over.
- **Fine-tuned NER model not in production.** The Bio_ClinicalBERT fine-tuning experiment (BC5CDR, F1 ≈ 0.84) surfaced a domain-shift issue — MRNs and surnames were misclassified as Chemical entities on real scanned hospital documents. It's kept as a documented experiment rather than deployed.
- **De-identified training data, not de-identification guarantees.** MTSamples is already de-identified; this project does not itself certify de-identification of arbitrary new documents fed into it.
- **Classifier accuracy is keyword/rule-based**, not a learned model, and has known ceiling effects after targeted keyword tuning.

---

## Roadmap

- [ ] Generalized patient name extraction in `lab_processor.py` (two-layer approach: broad field-label vocabulary + hospital-contact stop-boundaries, replacing single-hospital template patching)
- [ ] Evaluate BAA-compliant LLM hosting options to close the Groq compliance gap
- [ ] Revisit Bio_ClinicalBERT fine-tuning with domain-shift mitigation for MRN/surname disambiguation

---

## Deliverables

- 10-page project report (Word)
- 15-slide presentation deck (PowerPoint)
- Dockerized deployment

---

## License

Add your license here (e.g. MIT).
