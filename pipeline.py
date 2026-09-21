# 
"""
pipeline.py

Document ingestion & processing pipeline (Phases 1-6) for the
Intelligent Healthcare Document Assistant.

This module owns:
  - Folder setup
  - Streamlit session_state initialization
  - The end-to-end processing routine that runs on a new upload
    (text extraction -> cleaning -> classification -> NER ->
    structured extraction -> summarization -> RAG indexing ->
    risk/decision support)

IMPORTANT: this file is intentionally NOT named `streamlit.py`.
Naming it that would shadow the real `streamlit` package on
`import streamlit as st` and crash on import.
"""

import os
import hashlib

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

from src.nlp.summarizer import generate_summary

from src.extraction.pdf_processor import (
    extract_text_from_file,
    clean_text,
    save_uploaded_file,
    save_text_to_file
)

from src.processors.medical_extractor import (
    extract_spacy_entities,
    extract_medical_entities
)

from src.processors.entity_processor import (
    process_medical_entities
)

from src.extraction.document_classifier import (
    classify_document
)

from src.processors.discharge_processor import (
    extract_discharge_information
)

from src.processors.medicine_processor import (
    extract_medicine_information
)

from src.nlp.llm_extractor import (
    extract_lab_information_llm
)

from src.processors.radiology_processor import (
    extract_radiology_information
)

from src.processors.cardiology_processor import (
    extract_cardiology_information
)

from src.nlp.rag_engine import (
    index_document
)

from src.clinical.decision_support import (
    analyze_decision_support
)


# ============================================================
# NLTK SETUP
# ============================================================

def ensure_nltk_resources():
    """Download NLTK tokenizer resources if not already present."""

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"
TEXT_FOLDER = "extracted_text"
ANNOTATIONS_FOLDER = "annotations"


def ensure_folders():
    """Create the folders the pipeline writes to, if missing."""

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TEXT_FOLDER, exist_ok=True)
    os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    """Initialize every session_state key the app relies on."""

    defaults = {
        "qa_history": [],
        "processed": False,
        "document_hash": None,
        "document_text": "",
        "document_type": "",
        "structured_data": {},
        "summary": "",
        "risks": [],
        "overall_risk": "Low",
        "drug_interactions": [],
        "referrals": [],
        "report_pdf_bytes": None,
        "extraction_method": "",
        "spacy_entities": [],
        "processed_entities": [],
        "medical_entities": [],
        "translated_text": "",
        "translation_language": "",
        "translation_history": [],
        "raw_text": "",
        "word_count": 0,
        "character_count": 0,
        "sentence_count": 0,
        "text_file_path": "",
        "indexed": False,
        "primary_file_name": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_qa_history():
    """Clear previous Q&A history."""

    st.session_state.qa_history = []


def clear_translation():
    """Clear current translation."""

    st.session_state.translated_text = ""
    st.session_state.translation_language = ""


# ============================================================
# MAIN PROCESSING PIPELINE (Phases 1-6)
# ============================================================

def compute_upload_hash(uploaded_files):
    """Hash the combined bytes of every uploaded file."""

    combined_bytes = b"".join(f.getvalue() for f in uploaded_files)
    return hashlib.md5(combined_bytes).hexdigest()


def process_uploaded_document(uploaded_files, primary_file_name):
    """
    Run the full Phase 1-6 pipeline on a fresh upload and store
    every result in st.session_state, exactly as the original
    in-line block in app.py did.

    Returns True on success. On failure it surfaces an
    st.error/st.exception and calls st.stop() itself, matching
    the original behavior.
    """

    # ============================================================
    # PHASE 1 — extract text from EVERY uploaded file, in the
    # order they were uploaded, then join them into one combined
    # document.
    # ============================================================

    combined_text_parts = []
    extraction_methods_used = []

    with st.spinner(
        f"Extracting text from {len(uploaded_files)} file(s)..."
    ):

        for individual_file in uploaded_files:

            pdf_path = save_uploaded_file(individual_file, UPLOAD_FOLDER)

            page_text, page_method = extract_text_from_file(pdf_path)

            if page_text and page_text.strip():
                combined_text_parts.append(page_text)
                extraction_methods_used.append(page_method)

    raw_text = "\n\n--- Page Break ---\n\n".join(combined_text_parts)

    extraction_method = (
        ", ".join(sorted(set(extraction_methods_used)))
        if extraction_methods_used
        else "None"
    )

    st.session_state.extraction_method = extraction_method
    st.session_state.raw_text = raw_text

    if not raw_text or not raw_text.strip():
        st.error("❌ No text could be extracted from the uploaded file(s).")
        st.stop()

    # ============================================================
    # PHASE 2
    # ============================================================

    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        st.error("❌ Text became empty after cleaning.")
        st.stop()

    text_file_path = save_text_to_file(
        cleaned_text, primary_file_name, TEXT_FOLDER
    )

    st.session_state.text_file_path = text_file_path

    try:
        tokens = word_tokenize(cleaned_text)
    except Exception:
        tokens = cleaned_text.split()

    word_count = len(tokens)
    character_count = len(cleaned_text)
    sentence_count = len(
        [s for s in cleaned_text.split(".") if s.strip()]
    )

    st.session_state.word_count = word_count
    st.session_state.character_count = character_count
    st.session_state.sentence_count = sentence_count

    # ============================================================
    # PHASE 3
    # ============================================================

    with st.spinner("Classifying medical document..."):
        document_type = classify_document(cleaned_text)

    st.session_state.document_type = document_type

    with st.spinner("Extracting medical entities..."):
        spacy_entities = extract_spacy_entities(cleaned_text)

    with st.spinner("Running medical entity extraction..."):
        medical_entities = extract_medical_entities(cleaned_text)

    try:
        processed_entities = process_medical_entities(medical_entities)
    except Exception:
        processed_entities = medical_entities

    st.session_state.spacy_entities = spacy_entities
    st.session_state.medical_entities = medical_entities
    st.session_state.processed_entities = processed_entities

    structured_data = {}
    normalized_type = str(document_type).lower()

    if "discharge" in normalized_type or "summary" in normalized_type:
        structured_data = extract_discharge_information(cleaned_text)

    elif "general" in normalized_type or "medicine" in normalized_type:
        structured_data = extract_medicine_information(cleaned_text)

    elif "lab" in normalized_type or "laboratory" in normalized_type:
        structured_data = extract_lab_information_llm(cleaned_text)

    elif (
        "radiology" in normalized_type
        or "mri" in normalized_type
        or "ct" in normalized_type
    ):
        structured_data = extract_radiology_information(cleaned_text)

    elif "cardiology" in normalized_type or "cardiac" in normalized_type:
        structured_data = extract_cardiology_information(cleaned_text)

    else:
        structured_data = {}

    st.session_state.structured_data = structured_data

    # ============================================================
    # PHASE 4
    # ============================================================

    with st.spinner("Generating medical summary..."):
        summary = generate_summary(structured_data, document_type)

    if not summary:
        summary = "Summary could not be generated."

    st.session_state.summary = summary

    # ============================================================
    # PHASE 5 — RAG
    # ============================================================

    with st.spinner("Creating document embeddings and indexing..."):
        index_document(
            document_text=cleaned_text,
            document_name=primary_file_name,
            document_type=document_type,
            document_id=st.session_state.document_hash,
            clear_existing=True
        )

    st.session_state.indexed = True

    # ============================================================
    # PHASE 6 — RISK (Phase 12 wired in)
    # ============================================================

    with st.spinner("Analyzing clinical risk indicators..."):
        risk_result = analyze_decision_support(
            data=structured_data,
            document_type=document_type
        )

    if isinstance(risk_result, tuple):

        if len(risk_result) >= 2:
            risks = risk_result[0]
            overall_risk = risk_result[1]
        else:
            risks = risk_result
            overall_risk = "Low"

    elif isinstance(risk_result, dict):
        risks = risk_result.get("risks", [])
        overall_risk = risk_result.get("overall_risk", "Low")

    elif isinstance(risk_result, list):
        risks = risk_result

        priorities = [
            str(r.get("priority", r.get("severity", ""))).lower()
            for r in risks
            if isinstance(r, dict)
        ]

        if "high" in priorities:
            overall_risk = "High"
        elif "moderate" in priorities:
            overall_risk = "Moderate"
        else:
            overall_risk = "Low"

    else:
        risks = []
        overall_risk = "Low"

    st.session_state.risks = risks
    st.session_state.overall_risk = overall_risk

    if isinstance(risk_result, dict):
        st.session_state.drug_interactions = risk_result.get(
            "drug_interactions", []
        )
        st.session_state.referrals = risk_result.get("referrals", [])
    else:
        st.session_state.drug_interactions = []
        st.session_state.referrals = []

    # ============================================================
    # SAVE PROCESSING STATE
    # ============================================================

    st.session_state.document_text = cleaned_text
    st.session_state.processed = True

    return True


def handle_new_upload(uploaded_files):
    """
    Entry point called from app.py whenever the file_uploader
    has files in it. Detects whether this is a NEW document
    (by hash) and, if so, resets per-document state and runs
    the full pipeline. Mirrors the original inline logic in
    app.py exactly.
    """

    current_hash = compute_upload_hash(uploaded_files)

    if current_hash == st.session_state.document_hash:
        # Same document as last run - nothing to do.
        return

    primary_file_name = uploaded_files[0].name

    st.session_state.document_hash = current_hash
    st.session_state.processed = False
    st.session_state.primary_file_name = primary_file_name
    st.session_state.qa_history = []

    clear_translation()
    st.session_state.translation_history = []

    st.session_state.spacy_entities = []
    st.session_state.processed_entities = []
    st.session_state.medical_entities = []

    try:
        process_uploaded_document(uploaded_files, primary_file_name)

        st.success(
            "🎉 Document processing completed successfully! "
            "Use the sidebar to browse each phase's results."
        )

    except Exception as e:
        st.error("❌ An error occurred while processing the document.")
        st.exception(e)
        st.stop()