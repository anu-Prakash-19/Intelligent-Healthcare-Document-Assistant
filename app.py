import os
import tempfile

import streamlit as st
import pandas as pd
import plotly.express as px

from src.reports.report_generator import (
    generate_pdf_report
)

from src.storage.annotation_service import (
    add_annotation,
    get_annotations,
    delete_annotation,
    get_annotation_summary,
    build_section_options,
    FLAG_LEVELS
)

from src.search.search_service import (
    list_documents,
    delete_document,
    search_library_grouped
)

from src.nlp.rag_engine import (
    answer_with_rag
)

from src.nlp.translation_service import (
    translate_medical_document
)

from pipeline import (
    ensure_nltk_resources,
    ensure_folders,
    init_session_state,
    clear_qa_history,
    clear_translation,
    handle_new_upload,
)


# ============================================================
# STARTUP
# ============================================================

ensure_nltk_resources()
ensure_folders()

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Healthcare Document Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        color: #1f4e79;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .success-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }

    .warning-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
    }

    .danger-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }

    .info-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏥 Intelligent Healthcare Document Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered medical document processing, extraction, '
    'summarization, risk analysis, retrieval-based question '
    'answering and multilingual support.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

def display_patient_information(patient_info):
    """Display patient demographic information."""

    if not patient_info:
        st.info("No patient information extracted.")
        return

    st.markdown("### 👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Name", patient_info.get("name", "Not available"))

    with col2:
        age = patient_info.get("age", "Not available")
        st.metric("Age", str(age))

    with col3:
        st.metric("Sex", patient_info.get("sex", "Not available"))

    extra_data = {}

    if patient_info.get("mrn"):
        extra_data["MRN"] = patient_info.get("mrn")

    if patient_info.get("dob"):
        extra_data["DOB"] = patient_info.get("dob")

    if extra_data:
        st.dataframe(
            pd.DataFrame([extra_data]),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MEDICATIONS
# ============================================================

def display_medications(medications):
    """Display extracted medications."""

    st.markdown("### 💊 Medications")

    if not medications:
        st.info("No medications extracted.")
        return

    rows = []

    for medication in medications:

        if isinstance(medication, dict):

            rows.append(
                {
                    "Medication": medication.get("medication", ""),
                    "Dose": medication.get("dose", ""),
                    "Frequency": medication.get("frequency", ""),
                    "Route": medication.get("route", ""),
                    "Instructions": medication.get("instructions", ""),
                    "Status": medication.get("status", "")
                }
            )

        else:

            rows.append(
                {
                    "Medication": str(medication),
                    "Dose": "",
                    "Frequency": "",
                    "Route": "",
                    "Instructions": "",
                    "Status": ""
                }
            )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DIAGNOSES
# ============================================================

def display_diagnoses(diagnoses):
    """Display diagnosis list."""

    st.markdown("### 🩺 Diagnoses")

    if not diagnoses:
        st.info("No diagnoses extracted.")
        return

    for diagnosis in diagnoses:
        st.markdown(f"- **{diagnosis}**")


# ============================================================
# GENERIC SECTION
# ============================================================

def display_section(title, content):
    """Display generic section."""

    st.markdown(f"### {title}")

    if content is None:
        st.info("No information available.")
        return

    if isinstance(content, list):

        if len(content) == 0:
            st.info("No information available.")
            return

        for item in content:

            if isinstance(item, dict):
                st.json(item)
            else:
                st.markdown(f"- {item}")

    elif isinstance(content, dict):
        st.json(content)

    else:
        st.write(content)


# ============================================================
# PROCESSED ENTITIES
# ============================================================

def display_processed_entities(entities):
    """Display processed medical entities."""

    st.markdown("### 🧬 Medical Entities")

    if not entities:
        st.info("No medical entities detected.")
        return

    rows = []

    if isinstance(entities, dict):

        for label, values in entities.items():

            if isinstance(values, list):

                for value in values:
                    rows.append({"Entity": value, "Type": label})

            else:
                rows.append({"Entity": str(values), "Type": label})

    elif isinstance(entities, list):

        for entity in entities:

            if isinstance(entity, dict):

                rows.append(
                    {
                        "Entity": entity.get(
                            "text", entity.get("entity", "")
                        ),
                        "Type": entity.get(
                            "label", entity.get("type", "")
                        )
                    }
                )

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write(entities)


# ============================================================
# RISK COUNTS
# ============================================================

def get_risk_counts(risks):
    """Calculate risk counts."""

    high_count = 0
    moderate_count = 0
    low_count = 0

    for risk in risks:

        if not isinstance(risk, dict):
            continue

        priority = str(
            risk.get("priority", risk.get("severity", ""))
        ).lower()

        if priority == "high":
            high_count += 1
        elif priority == "moderate":
            moderate_count += 1
        elif priority == "low":
            low_count += 1

    return high_count, moderate_count, low_count


# ============================================================
# RISK DASHBOARD
# ============================================================

def display_risk_dashboard(risks, overall_risk):
    """Display risk dashboard."""

    st.markdown("## ⚠️ Risk Analysis")

    high_count, moderate_count, low_count = get_risk_counts(risks)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Overall Risk", overall_risk)

    with col2:
        st.metric("Total Risks", len(risks))

    with col3:
        st.metric("High Priority", high_count)

    with col4:
        st.metric("Moderate", moderate_count)

    if not risks:
        st.success("No rule-based risk indicators detected.")
        return

    risk_rows = []

    for risk in risks:

        if not isinstance(risk, dict):
            continue

        risk_rows.append(
            {
                "Risk": risk.get(
                    "category", risk.get("risk", risk.get("name", "Unknown"))
                ),
                "Priority": risk.get(
                    "priority", risk.get("severity", "Unknown")
                ),
                "Evidence": risk.get(
                    "finding", risk.get("evidence", "")
                ),
                "Recommendation": risk.get("recommendation", "")
            }
        )

    if not risk_rows:
        st.info("No structured risk information available.")
        return

    risk_df = pd.DataFrame(risk_rows)

    st.markdown("### Risk Details")

    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    priority_counts = (
        risk_df["Priority"]
        .astype(str)
        .str.title()
        .value_counts()
        .reset_index()
    )

    priority_counts.columns = ["Priority", "Count"]

    fig = px.bar(
        priority_counts,
        x="Priority",
        y="Count",
        title="Risk Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="risk_distribution_dashboard_chart"
    )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown("### 📋 Recommendations")

    for risk in risks:

        if not isinstance(risk, dict):
            continue

        priority = risk.get("priority", risk.get("severity", "Unknown"))

        risk_name = risk.get(
            "category", risk.get("risk", risk.get("name", "Risk"))
        )

        evidence = risk.get("finding", risk.get("evidence", ""))
        recommendation = risk.get("recommendation", "")

        if str(priority).lower() == "high":
            st.error(
                f"🔴 **{risk_name}**\n\n"
                f"Evidence: {evidence}\n\n"
                f"Recommendation: {recommendation}"
            )

        elif str(priority).lower() == "moderate":
            st.warning(
                f"🟠 **{risk_name}**\n\n"
                f"Evidence: {evidence}\n\n"
                f"Recommendation: {recommendation}"
            )

        else:
            st.info(
                f"🟢 **{risk_name}**\n\n"
                f"Evidence: {evidence}\n\n"
                f"Recommendation: {recommendation}"
            )


# ============================================================
# PHASE 12 — DRUG INTERACTIONS & GUIDELINE REFERRALS
# ============================================================

def display_decision_support_extras(drug_interactions, referrals):
    """
    Display Phase 12's two additions on top of Phase 6's risk
    dashboard: pairwise drug interaction findings and
    guideline-based specialty referral recommendations.
    """

    st.markdown("### 💊 Drug Interactions")

    if not drug_interactions:
        st.success(
            "No known interactions detected among the "
            "extracted medications."
        )

    else:

        for interaction in drug_interactions:

            if not isinstance(interaction, dict):
                continue

            severity = interaction.get("severity", "Unknown")
            finding = interaction.get("finding", "Drug interaction")
            explanation = interaction.get("explanation", "")
            recommendation = interaction.get("recommendation", "")

            message = (
                f"**{finding}**\n\n"
                f"{explanation}\n\n"
                f"Recommendation: {recommendation}"
            )

            if str(severity).lower() == "high":
                st.error(f"🔴 {message}")
            elif str(severity).lower() == "moderate":
                st.warning(f"🟠 {message}")
            else:
                st.info(f"🟢 {message}")

    st.markdown("### 🏥 Suggested Referrals")

    if not referrals:
        st.success(
            "No guideline-based referrals were triggered "
            "by this document's findings."
        )

    else:

        referral_rows = []

        for referral in referrals:

            if not isinstance(referral, dict):
                continue

            referral_rows.append(
                {
                    "Specialty": referral.get("specialty", "Unknown"),
                    "Urgency": referral.get("urgency", "Routine"),
                    "Reason": referral.get("reason", ""),
                    "Guideline Basis": referral.get("guideline_basis", "")
                }
            )

        if referral_rows:
            st.dataframe(
                pd.DataFrame(referral_rows),
                use_container_width=True,
                hide_index=True
            )

    st.caption(
        "⚠️ Decision support only — not a diagnosis. "
        "The interaction list is a small, illustrative set, "
        "not a substitute for a licensed drug-interaction "
        "checker (e.g. Lexicomp, Micromedex) in real clinical "
        "use."
    )


# ============================================================
# CLINICAL METRICS
# ============================================================

def display_clinical_metrics(data):
    """Display important clinical measurements."""

    st.markdown("### 🧪 Clinical Metrics")

    metrics = []

    if not isinstance(data, dict):
        return

    for key in ["hba1c", "bmi", "glucose", "egfr", "ldl", "lvef"]:

        value = data.get(key)

        if value:
            metrics.append((key.upper(), value))

    if not metrics:
        st.info("No clinical metrics available.")
        return

    columns = st.columns(min(len(metrics), 4))

    for index, (name, value) in enumerate(metrics):

        with columns[index % len(columns)]:

            if isinstance(value, dict):

                display_value = value.get("value", "")
                unit = value.get("unit", "")

                st.metric(name, f"{display_value} {unit}")

            else:
                st.metric(name, str(value))


# ============================================================
# NER DASHBOARD
# ============================================================

def display_ner_dashboard(entities):
    """Display NER dashboard."""

    st.markdown("## 🧬 Medical Entity Analysis")

    if not entities:
        st.info("No medical entities found.")
        return

    rows = []

    if isinstance(entities, list):

        for entity in entities:

            if isinstance(entity, dict):

                rows.append(
                    {
                        "Entity": entity.get(
                            "text", entity.get("entity", "")
                        ),
                        "Type": entity.get(
                            "label", entity.get("type", "")
                        )
                    }
                )

    elif isinstance(entities, dict):

        for label, values in entities.items():

            if isinstance(values, list):

                for value in values:
                    rows.append({"Entity": value, "Type": label})

            else:
                rows.append({"Entity": str(values), "Type": label})

    if not rows:
        st.write(entities)
        return

    entity_df = pd.DataFrame(rows)

    st.dataframe(
        entity_df,
        use_container_width=True,
        hide_index=True
    )

    type_counts = entity_df["Type"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]

    fig = px.bar(
        type_counts,
        x="Type",
        y="Count",
        title="Medical Entity Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="medical_entity_dashboard_chart"
    )


# ============================================================
# LAB DASHBOARD
# ============================================================

def display_lab_dashboard(lab_data):
    """Display laboratory dashboard."""

    st.markdown("## 🧪 Laboratory Analysis")

    if not lab_data:
        st.info("No laboratory information extracted.")
        return

    if isinstance(lab_data, dict):
        st.json(lab_data)
        return

    if isinstance(lab_data, list):

        rows = []

        for item in lab_data:

            if isinstance(item, dict):

                rows.append(
                    {
                        "Test": item.get("test", item.get("name", "")),
                        "Value": item.get("value", ""),
                        "Unit": item.get("unit", ""),
                        "Status": item.get("status", "")
                    }
                )

        if rows:

            lab_df = pd.DataFrame(rows)

            st.dataframe(
                lab_df,
                use_container_width=True,
                hide_index=True
            )

            if "Status" in lab_df.columns:

                status_counts = (
                    lab_df["Status"]
                    .replace("", "Normal/Unknown")
                    .value_counts()
                    .reset_index()
                )

                status_counts.columns = ["Status", "Count"]

                fig = px.bar(
                    status_counts,
                    x="Status",
                    y="Count",
                    title="Laboratory Result Status"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="laboratory_status_dashboard_chart"
                )

        else:
            st.write(lab_data)

    else:
        st.write(lab_data)


# ============================================================
# QA HISTORY
# ============================================================

def display_qa_history():
    """Display previous questions and answers."""

    if not st.session_state.qa_history:
        return

    st.markdown("### Previous Questions")

    for item in st.session_state.qa_history:

        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.markdown("---")


# ============================================================
# SIDEBAR — PROJECT NAVIGATOR (folder = phase, file = section)
# ============================================================

NAV_MAP = {
    "🏠 Home — Upload Document": [
        "Upload Medical Document"
    ],
    "Phase 1 — PDF Processing": [
        "Upload Status & Extraction Method",
        "Raw Extracted Text"
    ],
    "Phase 2 — Text Cleaning & Statistics": [
        "Cleaning Statistics",
        "Cleaned Text"
    ],
    "Phase 3 — Classification & Extraction": [
        "Document Classification",
        "spaCy & Medical Entities",
        "Structured Medical Information"
    ],
    "Phase 4 — Summarization": [
        "Medical Summary"
    ],
    "Phase 5 — Interactive Q&A (RAG)": [
        "Indexing Status",
        "Ask a Question"
    ],
    "Phase 6 — Risk & Decision Support": [
        "Overall Risk",
        "Risk Dashboard"
    ],
    "Phase 7 — Dashboard": [
        "Document Overview",
        "Processing Pipeline",
        "Document Information"
    ],
    "Phase 8 — OCR & Scanned Document Support": [
        "OCR Information"
    ],
    "Phase 9 — Multilingual Translation": [
        "Translate Document",
        "Translation Information"
    ],
    "Phase 10 — Document Library & Semantic Search": [
        "Indexed Documents",
        "Search Across All Documents"
    ],
    "Phase 12 — Enhanced Decision Support": [
        "Drug Interactions & Referrals"
    ],
    "Phase 13 — Downloadable PDF Reports": [
        "Generate & Download Report"
    ],
    "Phase 14 — Annotations & Flagging": [
        "Add Annotation",
        "Existing Annotations"
    ],
    "Raw Data": [
        "Raw & Structured Data"
    ]
}

st.sidebar.title("🏥 Healthcare Assistant")

st.sidebar.markdown("### 🗂️ Project Navigator")

nav_phase = st.sidebar.radio(
    "📁 Phase",
    list(NAV_MAP.keys()),
    key="nav_phase_select"
)

nav_file = st.sidebar.radio(
    "📄 File",
    NAV_MAP[nav_phase],
    key=f"nav_file_select_{nav_phase}"
)


# ============================================================
# MAIN AREA — UPLOAD WIDGET (always present, drives the
# whole pipeline regardless of which folder/file is selected
# in the sidebar). Accepts MULTIPLE files in one upload, so a
# multi-page paper document can be uploaded as several photos
# and combined into one document.
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📄 Upload Medical Document'
    '</div>',
    unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
    "Upload a medical document (single file, or multiple "
    "photos of one multi-page document)",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="medical_pdf_uploader"
)

st.caption(
    "📸 If uploading multiple photos of one multi-page "
    "document, select them in page order (page 1 first, "
    "page 2 second, etc.)."
)

if uploaded_files:
    handle_new_upload(uploaded_files)


# ============================================================
# NAVIGATION-DRIVEN CONTENT
# ============================================================

st.markdown("---")

NAV_NO_DOCUMENT_REQUIRED = {
    "🏠 Home — Upload Document",
    "Phase 8 — OCR & Scanned Document Support",
    "Phase 10 — Document Library & Semantic Search",
}

nav_requires_document = (
    nav_phase not in NAV_NO_DOCUMENT_REQUIRED
    and not (
        nav_phase == "Phase 9 — Multilingual Translation"
        and nav_file == "Translation Information"
    )
)

if nav_requires_document and not st.session_state.processed:

    st.info(
        "📭 No document has been processed yet. Upload a PDF "
        "above, then come back and browse this phase's results."
    )

else:

    # ========================================================
    # 🏠 HOME
    # ========================================================

    if nav_phase == "🏠 Home — Upload Document":

        if st.session_state.processed:
            st.success(
                "A document is currently loaded and processed. "
                "Browse its results using the sidebar folders."
            )
        else:
            st.info("Upload a medical PDF above to begin processing.")

    # ========================================================
    # PHASE 1 — PDF PROCESSING
    # ========================================================

    elif nav_phase == "Phase 1 — PDF Processing":

        st.header("📥 Phase 1: PDF Processing")

        if nav_file == "Upload Status & Extraction Method":

            extraction_method = st.session_state.extraction_method

            if extraction_method == "PDF Text + OCR":
                st.info(
                    "🔍 OCR was used because the PDF "
                    "did not contain enough machine-readable text."
                )

            elif extraction_method == "PDF Text":
                st.success("📄 Text extracted directly from the PDF.")

            elif extraction_method == "DOCX Text":
                st.success("📄 Text extracted from Word document.")

            elif extraction_method == "Image OCR":
                st.info("🔍 OCR was used to read the uploaded image.")

            else:
                st.warning("⚠️ No readable text was detected.")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Extraction Method", extraction_method)

            with col2:
                st.metric("Raw Characters", len(st.session_state.raw_text))

        elif nav_file == "Raw Extracted Text":

            with st.expander("👁️ View Extracted Text", expanded=True):
                st.text_area(
                    "Raw extracted text",
                    st.session_state.raw_text,
                    height=300,
                    key="raw_text_preview"
                )

    # ========================================================
    # PHASE 2 — TEXT CLEANING & STATISTICS
    # ========================================================

    elif nav_phase == "Phase 2 — Text Cleaning & Statistics":

        st.header("🧹 Phase 2: Text Cleaning & Statistics")

        if nav_file == "Cleaning Statistics":

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Characters", st.session_state.character_count)

            with col2:
                st.metric("Words", st.session_state.word_count)

            with col3:
                st.metric("Approx. Sentences", st.session_state.sentence_count)

            with col4:
                st.metric("Extraction", st.session_state.extraction_method)

            st.success(
                f"Cleaned text saved to: "
                f"{st.session_state.text_file_path}"
            )

        elif nav_file == "Cleaned Text":

            with st.expander("📝 View Cleaned Text", expanded=True):
                st.text_area(
                    "Cleaned text",
                    st.session_state.document_text,
                    height=350,
                    key="cleaned_text_preview"
                )

    # ========================================================
    # PHASE 3 — CLASSIFICATION & EXTRACTION
    # ========================================================

    elif nav_phase == "Phase 3 — Classification & Extraction":

        st.header("🔬 Phase 3: Document Classification & Extraction")

        if nav_file == "Document Classification":

            st.success(
                f"📑 Document Type: **{st.session_state.document_type}**"
            )

        elif nav_file == "spaCy & Medical Entities":

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 🔤 spaCy Entities")

                if st.session_state.spacy_entities:
                    st.write(st.session_state.spacy_entities)
                else:
                    st.info("No spaCy entities detected.")

            with col2:
                display_processed_entities(
                    st.session_state.processed_entities
                )

        elif nav_file == "Structured Medical Information":

            structured_data = st.session_state.structured_data

            st.markdown("### 📋 Structured Medical Information")

            if structured_data:

                patient_info = structured_data.get(
                    "patient_information", {}
                )

                if patient_info:
                    display_patient_information(patient_info)

                diagnoses = structured_data.get("diagnoses", [])

                if diagnoses:
                    display_diagnoses(diagnoses)

                medications = structured_data.get("medications", [])

                if medications:
                    display_medications(medications)

                display_clinical_metrics(structured_data)

                with st.expander("🔎 View Complete Structured JSON"):
                    st.json(structured_data)

            else:
                st.info(
                    "No structured information "
                    "was extracted for this document type."
                )

    # ========================================================
    # PHASE 4 — SUMMARIZATION
    # ========================================================

    elif nav_phase == "Phase 4 — Summarization":

        st.header("🧠 Phase 4: Medical Report Summarization")

        st.markdown("### 📄 Medical Summary")

        if st.session_state.summary:

            st.info(st.session_state.summary)

            st.download_button(
                label="⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="medical_summary.txt",
                mime="text/plain",
                key="download_summary_button"
            )

        else:
            st.info("No summary available.")

    # ========================================================
    # PHASE 5 — INTERACTIVE Q&A (RAG)
    # ========================================================

    elif nav_phase == "Phase 5 — Interactive Q&A (RAG)":

        st.header("💬 Phase 5: Interactive Medical Q&A")

        if nav_file == "Indexing Status":

            if st.session_state.indexed:
                st.success("✅ Document indexed successfully.")
            else:
                st.info("Document has not been indexed yet.")

        elif nav_file == "Ask a Question":

            st.info(
                "Ask questions only about the uploaded "
                "medical document. The RAG system retrieves "
                "relevant document sections before generating "
                "the answer."
            )

            question = st.text_input(
                "Enter your question",
                placeholder=(
                    "Example: Why is the patient's diabetes "
                    "considered poorly controlled?"
                ),
                key="rag_question_input"
            )

            col1, col2 = st.columns(2)

            with col1:
                ask_button = st.button("🔍 Ask Question", key="rag_ask_button")

            with col2:
                clear_button = st.button(
                    "🗑️ Clear History", key="rag_clear_button"
                )

            if clear_button:
                clear_qa_history()
                st.success("Q&A history cleared.")

            if ask_button:

                if not question.strip():
                    st.warning("Please enter a question.")

                else:

                    with st.spinner("Searching the medical document..."):

                        try:

                            answer = answer_with_rag(
                                question=question,
                                document_text=st.session_state.document_text,
                                top_k=5,
                                document_id=st.session_state.document_hash
                            )

                            if isinstance(answer, tuple):
                                final_answer = answer[0]

                            elif isinstance(answer, dict):
                                final_answer = answer.get("answer", str(answer))

                            else:
                                final_answer = str(answer)

                            st.session_state.qa_history.append(
                                {"question": question, "answer": final_answer}
                            )

                        except Exception as e:
                            st.error("❌ Error while generating answer.")
                            st.exception(e)

            display_qa_history()

    # ========================================================
    # PHASE 6 — RISK & DECISION SUPPORT
    # ========================================================

    elif nav_phase == "Phase 6 — Risk & Decision Support":

        st.header("⚠️ Phase 6: Risk & Decision Support")

        if nav_file == "Overall Risk":

            overall_risk = st.session_state.overall_risk

            if str(overall_risk).lower() == "high":
                st.error(f"🔴 Overall Risk: **{overall_risk}**")

            elif str(overall_risk).lower() == "moderate":
                st.warning(f"🟠 Overall Risk: **{overall_risk}**")

            else:
                st.success(f"🟢 Overall Risk: **{overall_risk}**")

        elif nav_file == "Risk Dashboard":

            display_risk_dashboard(
                st.session_state.risks, st.session_state.overall_risk
            )

    # ========================================================
    # PHASE 7 — DASHBOARD
    # ========================================================

    elif nav_phase == "Phase 7 — Dashboard":

        st.header("📊 Phase 7: Healthcare Dashboard")

        if nav_file == "Document Overview":

            st.subheader("🏠 Document Overview")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Document Type", st.session_state.document_type)

            with col2:
                st.metric("Extraction", st.session_state.extraction_method)

            with col3:
                st.metric("Characters", len(st.session_state.document_text))

            with col4:
                st.metric("Risk Level", st.session_state.overall_risk)

            annotation_summary_metrics = get_annotation_summary(
                st.session_state.document_hash
            )

            if annotation_summary_metrics["total"] > 0:
                st.info(
                    f"📌 {annotation_summary_metrics['total']} "
                    f"annotation(s) on this document — "
                    f"{annotation_summary_metrics['Critical']} Critical, "
                    f"{annotation_summary_metrics['Important']} Important."
                )

        elif nav_file == "Processing Pipeline":

            st.markdown("### 🔄 Processing Pipeline")

            pipeline_status = pd.DataFrame(
                {
                    "Phase": [
                        "PDF Upload",
                        "Text Extraction",
                        "Text Cleaning",
                        "Classification",
                        "NER",
                        "Structured Extraction",
                        "Summarization",
                        "RAG",
                        "Risk Analysis",
                        "Dashboard",
                        "OCR",
                        "Translation"
                    ],
                    "Status": [
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Completed",
                        "Available"
                    ]
                }
            )

            st.dataframe(
                pipeline_status,
                use_container_width=True,
                hide_index=True
            )

        elif nav_file == "Document Information":

            st.markdown("### 📌 Document Information")

            st.write(f"**Document Hash:** {st.session_state.document_hash}")
            st.write(f"**Type:** {st.session_state.document_type}")
            st.write(
                f"**Extraction Method:** "
                f"{st.session_state.extraction_method}"
            )

    # ========================================================
    # PHASE 8 — OCR INFORMATION
    # ========================================================

    elif nav_phase == "Phase 8 — OCR & Scanned Document Support":

        st.header("🔍 Phase 8: OCR & Scanned Document Support")

        st.markdown(
            """
            The application supports both:

            **1. Normal PDF**

            - Extracts machine-readable text from the PDF.

            **2. Scanned/Image PDF**

            - Detects when normal text extraction is insufficient.
            - Renders the PDF page as an image.
            - Uses Tesseract OCR to recognize the text.
            - Sends the extracted OCR text through the same NLP pipeline.

            **Processing flow:**

            **PDF → Text Extraction → OCR fallback → Cleaning →
            Classification → NER → Structured Extraction →
            Summary → RAG → Risk Analysis → Dashboard**
            """
        )

        st.info(
            "⚠️ OCR accuracy depends on scan quality, "
            "image resolution, font clarity and document layout."
        )

    # ========================================================
    # PHASE 9 — MULTILINGUAL TRANSLATION
    # ========================================================

    elif nav_phase == "Phase 9 — Multilingual Translation":

        st.header("🌐 Phase 9: Multilingual Medical Translation")

        if nav_file == "Translate Document":

            st.subheader("🌐 Phase 9: Multilingual Medical Translation")

            st.info(
                "Translate the medical document into another "
                "language while preserving important medical "
                "terms, medication names, dosages, laboratory "
                "values and clinical meaning."
            )

            st.markdown("### 🌍 Select Target Language")

            language_options = {
                "English": "English",
                "Hindi": "Hindi",
                "Kannada": "Kannada",
                "Tamil": "Tamil",
                "Telugu": "Telugu",
                "Malayalam": "Malayalam",
                "Marathi": "Marathi",
                "Bengali": "Bengali",
                "Gujarati": "Gujarati"
            }

            selected_language = st.selectbox(
                "Translate document to:",
                options=list(language_options.keys()),
                key="translation_language_select"
            )

            col1, col2 = st.columns(2)

            with col1:
                translate_button = st.button(
                    "🌐 Translate Document", key="translate_document_button"
                )

            with col2:
                clear_translation_button = st.button(
                    "🗑️ Clear Translation", key="clear_translation_button"
                )

            if clear_translation_button:
                clear_translation()
                st.success("Translation cleared.")

            if translate_button:

                source_text = st.session_state.document_text

                if not source_text.strip():
                    st.warning("No document text is available for translation.")

                else:

                    with st.spinner(
                        f"Translating medical document to "
                        f"{selected_language}..."
                    ):

                        try:

                            translated_text = translate_medical_document(
                                text=source_text,
                                target_language=selected_language
                            )

                            if translated_text:

                                st.session_state.translated_text = translated_text
                                st.session_state.translation_language = (
                                    selected_language
                                )

                                st.session_state.translation_history.append(
                                    {
                                        "language": selected_language,
                                        "text": translated_text
                                    }
                                )

                                st.success(
                                    f"✅ Translation completed "
                                    f"in {selected_language}."
                                )

                            else:
                                st.warning("No translated text was returned.")

                        except Exception as e:
                            st.error(
                                "❌ Error while translating "
                                "the medical document."
                            )
                            st.exception(e)

            if st.session_state.translated_text:

                st.markdown(
                    f"### 📄 Translation "
                    f"({st.session_state.translation_language})"
                )

                st.text_area(
                    "Translated Medical Document",
                    st.session_state.translated_text,
                    height=500,
                    key="translated_document_display"
                )

                st.download_button(
                    label="⬇️ Download Translation",
                    data=st.session_state.translated_text,
                    file_name="translated_medical_document.txt",
                    mime="text/plain",
                    key="download_translation_button"
                )

            else:
                st.info(
                    "Select a language and click "
                    "'Translate Document' to generate "
                    "a translation."
                )

        elif nav_file == "Translation Information":

            st.markdown(
                """
                Phase 9 adds multilingual support to the healthcare
                document assistant.

                **Translation flow:**

                **Medical PDF → OCR/Text Extraction → Cleaning →
                Medical Processing → Translation → Translated Output**

                Supported target languages currently include:

                - English
                - Hindi
                - Kannada
                - Tamil
                - Telugu
                - Malayalam
                - Marathi
                - Bengali
                - Gujarati

                The translation system is designed to preserve:

                - Medication names
                - Dosages
                - Laboratory values
                - Units
                - Medical terminology
                - Dates
                - Clinical meaning
                """
            )

            st.warning(
                "⚠️ Translation is intended for development/educational "
                "use. Medical translations should be reviewed by a "
                "qualified healthcare professional before clinical use."
            )

    # ========================================================
    # PHASE 10 — DOCUMENT LIBRARY & SEMANTIC SEARCH
    # ========================================================

    elif nav_phase == "Phase 10 — Document Library & Semantic Search":

        st.header("📚 Phase 10: Document Library & Semantic Search")

        indexed_documents = list_documents()

        if nav_file == "Indexed Documents":

            st.markdown("#### Indexed Documents")

            if not indexed_documents:
                st.info(
                    "No documents are indexed yet. Upload a "
                    "document above to add it to the library."
                )

            else:

                library_rows = []

                for doc in indexed_documents:

                    library_rows.append(
                        {
                            "Document": doc.get("document", "Unknown"),
                            "Type": doc.get("document_type", "Unknown"),
                            "Chunks": doc.get("chunk_count", 0),
                            "Indexed At": doc.get("indexed_at", ""),
                            "Document ID": doc.get("document_id", "")
                        }
                    )

                library_df = pd.DataFrame(library_rows)

                st.dataframe(
                    library_df.drop(columns=["Document ID"]),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("#### Remove a Document")

                document_labels = {
                    f"{doc.get('document', 'Unknown')} "
                    f"({doc.get('document_type', 'Unknown')})": doc.get(
                        "document_id"
                    )
                    for doc in indexed_documents
                }

                selected_label = st.selectbox(
                    "Select a document to remove from the library",
                    options=list(document_labels.keys()),
                    key="library_delete_select"
                )

                if st.button(
                    "🗑️ Remove Selected Document", key="library_delete_button"
                ):

                    selected_document_id = document_labels[selected_label]

                    delete_result = delete_document(selected_document_id)

                    if delete_result.get("success"):
                        st.success(
                            f"Removed '{selected_label}' from the library."
                        )
                        st.rerun()

                    else:
                        st.error(
                            delete_result.get(
                                "message", "Could not remove the document."
                            )
                        )

        elif nav_file == "Search Across All Documents":

            st.markdown("#### Search Across All Documents")

            st.caption(
                "Unlike the Q&A above (which only answers from the "
                "single report currently open), this searches every "
                "document ever indexed into the library."
            )

            available_types = sorted(
                {
                    doc.get("document_type")
                    for doc in indexed_documents
                    if doc.get("document_type")
                }
            )

            type_filter_options = ["All types"] + available_types

            selected_type_filter = st.selectbox(
                "Filter by document type (optional)",
                options=type_filter_options,
                key="library_search_type_filter"
            )

            library_query = st.text_input(
                "Search query",
                placeholder=(
                    "Example: elevated glucose, meningioma, "
                    "discharge medications"
                ),
                key="library_search_query"
            )

            if st.button("🔎 Search Library", key="library_search_button"):

                if not library_query.strip():
                    st.warning("Please enter a search query.")

                elif not indexed_documents:
                    st.warning("The library is empty. Upload documents first.")

                else:

                    document_type_filter = (
                        None
                        if selected_type_filter == "All types"
                        else selected_type_filter
                    )

                    with st.spinner("Searching the document library..."):

                        grouped_results = search_library_grouped(
                            query=library_query,
                            top_k=10,
                            document_type=document_type_filter
                        )

                    if not grouped_results:
                        st.info("No matching results were found.")

                    else:

                        st.markdown(
                            f"**Found matches in "
                            f"{len(grouped_results)} document(s):**"
                        )

                        for group in grouped_results:

                            header = (
                                f"📄 {group.get('document', 'Unknown')} "
                                f"— {group.get('document_type', 'Unknown')} "
                                f"(best match score: "
                                f"{group.get('best_score', 0):.3f})"
                            )

                            with st.expander(header):

                                for match in group.get("matches", []):

                                    st.markdown(
                                        f"**Score:** "
                                        f"{match.get('score', 0):.3f} "
                                        f"&nbsp;|&nbsp; "
                                        f"**Page:** "
                                        f"{match.get('page', 'N/A')}"
                                    )

                                    st.write(match.get("text", ""))

                                    st.markdown("---")

    # ========================================================
    # PHASE 12 — ENHANCED DECISION SUPPORT
    # ========================================================

    elif nav_phase == "Phase 12 — Enhanced Decision Support":

        st.header("💊 Phase 12: Enhanced Decision Support")

        display_decision_support_extras(
            st.session_state.drug_interactions,
            st.session_state.referrals
        )

    # ========================================================
    # PHASE 13 — DOWNLOADABLE PDF REPORTS
    # ========================================================

    elif nav_phase == "Phase 13 — Downloadable PDF Reports":

        st.header("📄 Phase 13: Downloadable PDF Reports")

        st.markdown("### 📄 Downloadable Report")

        st.caption(
            "Generate a single PDF combining the patient "
            "summary, extracted clinical data, and risk & "
            "decision support findings for this document."
        )

        if st.button("📄 Generate PDF Report", key="generate_pdf_report_button"):

            with st.spinner("Building PDF report..."):

                report_path = os.path.join(
                    tempfile.gettempdir(),
                    f"medical_report_{st.session_state.document_hash}.pdf"
                )

                generate_pdf_report(
                    output_path=report_path,
                    document_name=st.session_state.primary_file_name,
                    document_type=st.session_state.document_type,
                    extraction_method=st.session_state.extraction_method,
                    summary=st.session_state.summary,
                    structured_data=st.session_state.structured_data,
                    risks=st.session_state.risks,
                    overall_risk=st.session_state.overall_risk,
                    drug_interactions=st.session_state.drug_interactions,
                    referrals=st.session_state.referrals
                )

                with open(report_path, "rb") as report_file:
                    st.session_state.report_pdf_bytes = report_file.read()

            st.success("✅ Report generated. Click below to download.")

        if st.session_state.get("report_pdf_bytes"):

            report_filename = (
                f"medical_report_{st.session_state.document_type}.pdf"
            ).replace(" ", "_")

            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state.report_pdf_bytes,
                file_name=report_filename,
                mime="application/pdf",
                key="download_pdf_report_button"
            )

    # ========================================================
    # PHASE 14 — ANNOTATIONS & FLAGGING
    # ========================================================

    elif nav_phase == "Phase 14 — Annotations & Flagging":

        st.header("📌 Phase 14: Annotations & Flagging")

        document_id = st.session_state.document_hash

        annotation_summary = get_annotation_summary(document_id)

        if nav_file == "Add Annotation":

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Annotations", annotation_summary["total"])

            with col2:
                st.metric("🔴 Critical", annotation_summary["Critical"])

            with col3:
                st.metric("🟠 Important", annotation_summary["Important"])

            with col4:
                st.metric("🟢 Notes", annotation_summary["Note"])

            st.markdown("---")

            st.markdown("### ➕ Add Annotation")

            section_options = build_section_options(
                st.session_state.structured_data,
                st.session_state.risks
            )

            selected_section = st.selectbox(
                "What does this note relate to?",
                options=section_options,
                key="annotation_section_select"
            )

            selected_flag_level = st.selectbox(
                "Flag level",
                options=FLAG_LEVELS,
                key="annotation_flag_level_select"
            )

            note_text = st.text_area(
                "Note",
                placeholder=(
                    "Example: Verify this dosage with the "
                    "prescribing physician before follow-up."
                ),
                key="annotation_note_input"
            )

            if st.button("📌 Add Annotation", key="add_annotation_button"):

                if not note_text.strip():
                    st.warning("Please enter a note before saving.")

                else:

                    add_annotation(
                        document_id=document_id,
                        section=selected_section,
                        note=note_text,
                        flag_level=selected_flag_level
                    )

                    st.success("Annotation added.")
                    st.rerun()

        elif nav_file == "Existing Annotations":

            st.markdown("### 📋 Existing Annotations")

            existing_annotations = get_annotations(document_id)

            if not existing_annotations:
                st.info("No annotations yet for this document.")

            else:

                flag_icons = {
                    "Critical": "🔴",
                    "Important": "🟠",
                    "Note": "🟢"
                }

                for annotation in existing_annotations:

                    icon = flag_icons.get(annotation.get("flag_level"), "🟢")

                    with st.container():

                        st.markdown(
                            f"{icon} **{annotation.get('flag_level', 'Note')}** "
                            f"— *{annotation.get('section', 'General')}*"
                        )

                        st.write(annotation.get("note", ""))
                        st.caption(annotation.get("created_at", ""))

                        if st.button(
                            "🗑️ Delete",
                            key=f"delete_annotation_{annotation.get('id')}"
                        ):

                            delete_annotation(document_id, annotation.get("id"))
                            st.rerun()

                        st.markdown("---")

    # ========================================================
    # RAW DATA
    # ========================================================

    elif nav_phase == "Raw Data":

        st.header("🗂️ Raw & Structured Data")

        st.markdown("### Cleaned Document Text")

        st.text_area(
            "Document",
            st.session_state.document_text,
            height=400,
            key="raw_dashboard_text"
        )

        st.markdown("### Structured Extraction")
        st.json(st.session_state.structured_data)

        st.markdown("### Risk Data")
        st.json(
            {
                "overall_risk": st.session_state.overall_risk,
                "risks": st.session_state.risks
            }
        )

        st.markdown("### Medical Entities")
        st.json(st.session_state.processed_entities)

        st.download_button(
            label="⬇️ Download Extracted Text",
            data=st.session_state.document_text,
            file_name="extracted_medical_text.txt",
            mime="text/plain",
            key="download_extracted_text_button"
        )