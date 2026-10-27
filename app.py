"""
=============================================================================
PROJECT: AI-Powered Universal Data Analyst (Power BI & Neobrutalism Edition)
Universal Ingestion | Auto-Cleaning | Power BI Dashboard | AutoML | Groq Voice Chatbot
=============================================================================
Main Application Entry Point (app.py)
"""

import os
import io
import json
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

# Internal modular engines
from src.loader import (
    load_tabular_from_bytes,
    save_raw_file,
    compute_dataset_summary,
    extract_zip_to_raw,
    scan_dataset_directory,
    load_and_append_tables,
)
from src.cleaner import (
    execute_full_cleaning_pipeline,
    get_downloadable_bytes,
)
from src.eda import (
    generate_correlation_heatmap,
    generate_distribution_chart,
    generate_time_trend_chart,
    build_custom_chart,
    generate_powerbi_donut,
    generate_powerbi_clustered_bar,
    generate_powerbi_area,
    generate_powerbi_treemap,
    generate_powerbi_matrix,
    generate_powerbi_gauge,
)
from src.splitter import (
    split_tabular_data,
    split_image_dataset,
    create_split_distribution_chart,
)
from src.automl import (
    infer_task_type,
    train_tabular_models,
    predict_tabular_sample,
)
from src.image_trainer import (
    train_mobilenet_v2,
    predict_single_image,
    create_training_curves_chart,
    create_confusion_matrix_chart,
)
from src.chatbot import (
    generate_groq_pandas_code,
    execute_safe_query,
    transcribe_audio_groq,
    generate_tts_audio,
    offline_fallback_analyst,
    SUPPORTED_GROQ_MODELS,
    WHISPER_MODELS,
)
from src.reporter import (
    generate_pdf_report,
    generate_word_report,
    generate_json_session_export,
)
from src.samples import ensure_sample_datasets

# Try importing mic recorder for voice input in chatbot
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False

# Ensure sample datasets exist
SAMPLE_PATHS = ensure_sample_datasets()

# Page configuration
st.set_page_config(
    page_title="DataFlow | Universal AI Analyst",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",  # No old sidebar buttons!
)

# -----------------------------------------------------------------------------
# HIGH-CONTRAST NEOBRUTALISM & POWER BI HYBRID DESIGN SYSTEM
# -----------------------------------------------------------------------------
NEOBRUTALISM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

    /* Global Base */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #f7f6f0 !important;
        color: #111827 !important;
    }

    /* Universal High-Contrast Text Enforcement */
    p, span, label, h1, h2, h3, h4, h5, h6, li, td, th {
        color: #111827 !important;
    }

    /* Widget Labels */
    label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p, .stWidgetLabel {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
    }

    /* Captions & Subtitles */
    .stCaption, [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }

    /* Streamlit Tabs - Modern Neobrutalist Capsules */
    div[data-testid="stTabs"] {
        margin-top: 12px;
        margin-bottom: 22px;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 9999px !important;
        padding: 8px 22px !important;
        margin-right: 12px !important;
        box-shadow: 3px 3px 0px #000000 !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        transform: translate(-1px, -1px);
        box-shadow: 4px 4px 0px #000000 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"] p,
    div[data-testid="stTabs"] button[data-baseweb="tab"] div,
    div[data-testid="stTabs"] button[data-baseweb="tab"] span {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #bef264 !important;
        border: 2.5px solid #000000 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {
        color: #000000 !important;
        font-weight: 900 !important;
    }
    div[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] div[data-baseweb="tab-border"] {
        display: none !important;
    }

    /* File Uploader Container - High Contrast */
    div[data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2.5px solid #000000 !important;
        border-radius: 18px !important;
        box-shadow: 4px 4px 0px #000000 !important;
        padding: 16px !important;
        margin-top: 10px !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stFileUploader"] section {
        background: #fafaf9 !important;
        border: 2px dashed #000000 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section small {
        color: #111827 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stFileUploader"] button {
        background: #8b5cf6 !important;
        color: #ffffff !important;
        border: 2px solid #000000 !important;
        box-shadow: 2px 2px 0px #000000 !important;
    }
    div[data-testid="stFileUploader"] button p {
        color: #ffffff !important;
    }

    /* Radio & Checkbox Labels */
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] p,
    div[data-testid="stCheckbox"] label, div[data-testid="stCheckbox"] p {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Top Floating Capsule Bar */
    .top-capsule-bar {
        background: #ffffff;
        border: 2.5px solid #000000;
        border-radius: 9999px;
        box-shadow: 4px 4px 0px #000000;
        padding: 8px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 25px;
    }

    /* Neo Buttons */
    .stButton > button {
        border: 2px solid #000000 !important;
        border-radius: 9999px !important;
        font-weight: 800 !important;
        font-size: 0.85rem !important;
        box-shadow: 3px 3px 0px #000000 !important;
        transition: all 0.12s ease !important;
        color: #000000 !important;
        background-color: #ffffff !important;
        padding: 6px 18px !important;
    }
    .stButton > button:hover {
        transform: translate(-1px, -1px) !important;
        box-shadow: 4px 4px 0px #000000 !important;
        border-color: #000000 !important;
        color: #000000 !important;
    }
    .stButton > button:active {
        transform: translate(2px, 2px) !important;
        box-shadow: 1px 1px 0px #000000 !important;
    }

    /* Neo Cards with Hard Offset Shadow */
    .neo-card {
        background: #ffffff;
        border: 2.5px solid #000000;
        border-radius: 20px;
        box-shadow: 4px 4px 0px #000000;
        padding: 22px;
        margin-bottom: 20px;
        transition: transform 0.15s ease;
    }
    .neo-card:hover {
        transform: translateY(-2px);
    }
    .neo-card-title {
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569 !important;
        margin-bottom: 6px;
    }
    .neo-card-val {
        font-size: 2rem;
        font-weight: 900;
        color: #000000 !important;
    }

    /* Power BI Style KPI Scorecard */
    .pbi-scorecard {
        background: #ffffff;
        border: 2px solid #000000;
        border-radius: 14px;
        box-shadow: 3px 3px 0px #000000;
        padding: 14px 18px;
        margin-bottom: 15px;
    }
    .pbi-scorecard-label {
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
    }
    .pbi-scorecard-num {
        font-size: 1.85rem;
        font-weight: 900;
        color: #0078D4;
        margin-top: 4px;
    }

    /* Pill Badges */
    .neo-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border: 2px solid #000000;
        border-radius: 9999px;
        padding: 4px 14px;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-purple { background-color: #c084fc !important; color: #000000 !important; }
    .badge-lime { background-color: #bef264 !important; color: #000000 !important; }
    .badge-yellow { background-color: #fef08a !important; color: #000000 !important; }
    .badge-cyan { background-color: #a5f3fc !important; color: #000000 !important; }
    .badge-dark { background-color: #000000 !important; color: #ffffff !important; }
    .badge-dark span { color: #ffffff !important; }

    /* Hero AI Copilot Card */
    .ai-copilot-card {
        background: #0f172a !important;
        border: 3px solid #bef264 !important;
        border-radius: 22px !important;
        box-shadow: 6px 6px 0px #000000 !important;
        padding: 22px !important;
    }
    .ai-copilot-card, .ai-copilot-card p, .ai-copilot-card div, .ai-copilot-card span {
        color: #ffffff !important;
    }
    .ai-copilot-card .ai-copilot-sub {
        color: #cbd5e1 !important;
        font-size: 0.85rem;
        line-height: 1.45;
        margin-bottom: 14px;
    }
    .ai-copilot-pill {
        background: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 9999px !important;
        padding: 8px 14px !important;
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 0.82rem !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .ai-copilot-pill span, .ai-copilot-pill p {
        color: #000000 !important;
        font-weight: 800 !important;
    }

    /* Milestone Cards at Bottom */
    .milestone-card {
        background: #ffffff;
        border: 2px solid #000000;
        border-radius: 16px;
        box-shadow: 4px 4px 0px #000000;
        padding: 16px;
        height: 100%;
        cursor: pointer;
        transition: transform 0.12s ease;
    }
    .milestone-card:hover {
        transform: translateY(-2px);
    }
    .milestone-card div {
        color: #000000 !important;
    }

    /* Text Inputs and Selectboxes */
    .stTextInput > div > div > input, .stSelectbox > div > div {
        border: 2px solid #000000 !important;
        border-radius: 12px !important;
        box-shadow: 2px 2px 0px #000000 !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Dataframe wrapper */
    .stDataFrame {
        border: 2px solid #000000 !important;
        border-radius: 14px !important;
        box-shadow: 3px 3px 0px #000000 !important;
        overflow: hidden !important;
        background-color: #ffffff !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 16px !important;
        box-shadow: 3px 3px 0px #000000 !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span {
        color: #000000 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 2px solid #000000 !important;
        border-radius: 14px !important;
        box-shadow: 3px 3px 0px #000000 !important;
    }
    [data-testid="stExpander"] p, [data-testid="stExpander"] span {
        color: #000000 !important;
    }
</style>
"""
st.markdown(NEOBRUTALISM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global State Initializer
# -----------------------------------------------------------------------------
def init_session():
    defaults = {
        "raw_df": None,
        "cleaned_df": None,
        "raw_filename": None,
        "dataset_type": None,
        "cleaning_log": [],
        "image_data_items": [],
        "image_classes": {},
        "image_splits": None,
        "image_model_results": None,
        "tabular_splits": None,
        "ml_target": None,
        "ml_results": None,
        "chat_history": [],
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "selected_groq_model": "qwen/qwen3.8-27b",
        "selected_whisper_model": "whisper-large-v3-turbo",
        "active_module": "ingest",  # 'ingest', 'clean', 'eda', 'automl', 'chat', 'report'
        "language": "English",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# -----------------------------------------------------------------------------
# TOP FLOATING CAPSULE NAVIGATION (NO OLD SIDEBAR BUTTONS!)
# -----------------------------------------------------------------------------
col_logo, col_nav1, col_nav2, col_nav3, col_nav4, col_nav5, col_nav6, col_badge = st.columns(
    [2.2, 1.3, 1.3, 1.4, 1.3, 1.5, 1.3, 2.0]
)

with col_logo:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; padding-top: 4px;">
            <div style="background: #8b5cf6; width: 34px; height: 34px; border: 2px solid #000; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; color: #fff; font-weight: 900; box-shadow: 2px 2px 0px #000;">
                ✦
            </div>
            <div>
                <span style="font-weight: 900; font-size: 1.25rem; color: #000; letter-spacing: -0.02em;">DataFlow</span>
                <span class="neo-badge badge-purple" style="font-size: 0.65rem; padding: 2px 8px; margin-left: 4px;">Power BI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav1:
    btn_label = "📁 Ingest" if st.session_state.active_module != "ingest" else "👉 Ingest"
    if st.button(btn_label, use_container_width=True):
        st.session_state.active_module = "ingest"
        st.rerun()

with col_nav2:
    btn_label = "🧹 Clean" if st.session_state.active_module != "clean" else "👉 Clean"
    if st.button(btn_label, use_container_width=True):
        st.session_state.active_module = "clean"
        st.rerun()

with col_nav3:
    btn_label = "📊 Power BI" if st.session_state.active_module != "eda" else "👉 Power BI"
    if st.button(btn_label, use_container_width=True):
        st.session_state.active_module = "eda"
        st.rerun()

with col_nav4:
    btn_label = "🤖 AutoML" if st.session_state.active_module != "automl" else "👉 AutoML"
    if st.button(btn_label, use_container_width=True):
        st.session_state.active_module = "automl"
        st.rerun()

with col_nav5:
    btn_label = "💬 Voice & Text AI" if st.session_state.active_module != "chat" else "👉 Voice & Text AI"
    if st.button(btn_label, use_container_width=True):
        st.session_state.active_module = "chat"
        st.rerun()

with col_nav6:
    btn_label = "📑 Reports" if st.session_state.active_module != "report" else "👉 Reports"
    if st.button(btn_label, use_container_width=True):
        st.session_state.active_module = "report"
        st.rerun()

with col_badge:
    engine_status = "⚡ GROQ ENGINE ACTIVE" if st.session_state.groq_api_key else "⚡ LOCAL AI ACTIVE"
    badge_bg = "badge-lime" if st.session_state.groq_api_key else "badge-yellow"
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 5px;">
            <span class="neo-badge {badge_bg}">{engine_status}</span>
            <span class="neo-badge badge-dark">ADMIN: Viva</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border: 1.5px solid #000; margin-top: 8px; margin-bottom: 24px;'>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# HERO SECTION WITH UNIFIED AI DATA COPILOT
# -----------------------------------------------------------------------------
hero_left, hero_right = st.columns([1.3, 1.4])

with hero_left:
    st.markdown(
        """
        <div style="font-size: 2.5rem; font-weight: 900; color: #000000; letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 10px;">
            Seamless <span style="text-decoration: underline; text-decoration-color: #8b5cf6;">Data</span> Journey ⤳
        </div>
        <div style="font-size: 1.02rem; color: #475569; font-weight: 600; line-height: 1.5; margin-bottom: 18px; max-width: 580px;">
            Universal AI Data Analyst: Autonomous Cleaning, Power BI Dashboard, AutoML Leaderboards, 
            and Multilingual Voice & Text Chatbot in English & Tamil.
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2 = st.columns([1.5, 1.5])
    with btn_c1:
        if st.button("⚡ Test With Titanic (1-Click)", key="hero_btn_titanic", use_container_width=True):
            df_t = pd.read_csv(SAMPLE_PATHS["titanic"])
            st.session_state.raw_df = df_t
            st.session_state.cleaned_df = df_t.copy()
            st.session_state.raw_filename = "titanic.csv (Sample)"
            st.session_state.dataset_type = "tabular"
            st.session_state.tabular_splits = None
            st.session_state.ml_results = None
            st.rerun()
    with btn_c2:
        if st.button("📁 Load Custom Dataset", key="hero_btn_ingest", use_container_width=True):
            st.session_state.active_module = "ingest"
            st.rerun()

    st.markdown(
        """
        <div style="margin-top: 14px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <span style="font-weight: 800; font-size: 0.78rem; text-transform: uppercase;">Universal Formats:</span>
            <span class="neo-badge badge-purple">CSV / TSV</span>
            <span class="neo-badge badge-lime">Excel (.xlsx/.xls)</span>
            <span class="neo-badge badge-cyan">Parquet & JSON</span>
            <span class="neo-badge badge-yellow">Vision Folders</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        """
        <div style="background: #ffffff; border: 2.5px solid #000000; border-radius: 20px; box-shadow: 4px 4px 0px #000000; padding: 20px 22px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 6px;">
                <span class="neo-badge badge-purple" style="font-size: 0.68rem;">SYSTEM ARCHITECTURE</span>
                <span class="neo-badge badge-lime" style="font-size: 0.68rem;">END-TO-END PIPELINE</span>
                <span class="neo-badge badge-cyan" style="font-size: 0.68rem;">ENTERPRISE VIVA READY</span>
            </div>
            
            <div style="font-size: 1.25rem; font-weight: 900; color: #000000; line-height: 1.2; margin-bottom: 6px;">
                ✦ Universal AI Data Analyst — System Blueprint
            </div>
            <div style="font-size: 0.82rem; color: #475569; font-weight: 600; line-height: 1.45; margin-bottom: 14px;">
                Fully automated analytics engine: from multi-format raw data ingestion to Power BI executive dashboards, dual AutoML, and multilingual conversational AI.
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px; margin-bottom: 14px;">
                <div style="background: #f8fafc; border: 1.5px solid #000000; border-radius: 12px; padding: 10px 12px;">
                    <div style="font-weight: 900; font-size: 0.82rem; color: #000000; margin-bottom: 3px;">
                        📁 1. Universal Ingest
                    </div>
                    <div style="font-size: 0.74rem; color: #334155; line-height: 1.35; font-weight: 600;">
                        Auto delimiter sniff (CSV/TSV), multi-sheet Excel, Parquet, JSON & recursive vision folders.
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1.5px solid #000000; border-radius: 12px; padding: 10px 12px;">
                    <div style="font-weight: 900; font-size: 0.82rem; color: #000000; margin-bottom: 3px;">
                        🧼 2. Autonomous Cleaner
                    </div>
                    <div style="font-size: 0.74rem; color: #334155; line-height: 1.35; font-weight: 600;">
                        Type coercion, median/mode imputation, IQR outlier capping & zero data-leakage splits.
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1.5px solid #000000; border-radius: 12px; padding: 10px 12px;">
                    <div style="font-weight: 900; font-size: 0.82rem; color: #000000; margin-bottom: 3px;">
                        📊 3. Power BI Dashboard
                    </div>
                    <div style="font-size: 0.74rem; color: #334155; line-height: 1.35; font-weight: 600;">
                        Interactive Slicers, 5 KPI Scorecards, Donut decomposition, Treemap & dynamic Matrix table.
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1.5px solid #000000; border-radius: 12px; padding: 10px 12px;">
                    <div style="font-weight: 900; font-size: 0.82rem; color: #000000; margin-bottom: 3px;">
                        🤖 4. Dual AutoML Studio
                    </div>
                    <div style="font-size: 0.74rem; color: #334155; line-height: 1.35; font-weight: 600;">
                        Scikit-Learn tabular pipelines (RF, GB, Ridge) with 5-fold CV + PyTorch MobileNetV2 for Vision.
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1.5px solid #000000; border-radius: 12px; padding: 10px 12px;">
                    <div style="font-weight: 900; font-size: 0.82rem; color: #000000; margin-bottom: 3px;">
                        🎙️ 5. Voice & Text Copilot
                    </div>
                    <div style="font-size: 0.74rem; color: #334155; line-height: 1.35; font-weight: 600;">
                        Groq LLM Pandas code generator, AST safety sandbox, Whisper STT mic & English/தமிழ் queries.
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1.5px solid #000000; border-radius: 12px; padding: 10px 12px;">
                    <div style="font-weight: 900; font-size: 0.82rem; color: #000000; margin-bottom: 3px;">
                        📑 6. Executive Reports
                    </div>
                    <div style="font-size: 0.74rem; color: #334155; line-height: 1.35; font-weight: 600;">
                        1-click comprehensive exports in PDF (ReportLab), Word (.docx), and session JSON audit logs.
                    </div>
                </div>
            </div>

            <div style="background: #fef9c3; border: 1.5px solid #000000; border-radius: 12px; padding: 8px 12px; display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.1rem;">🎓</span>
                <span style="font-size: 0.74rem; color: #000000; font-weight: 700; line-height: 1.3;">
                    <b>Viva Voce Defense Ready:</b> Zero data leakage with isolated splits, Python AST sandboxing against malicious queries, and offline fallback resilience.
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='border: 1.5px solid #000; margin-top: 25px; margin-bottom: 25px;'>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# QUICK MILESTONE TILES (ALL 6 CAPABILITIES TOGETHER)
# -----------------------------------------------------------------------------
st.markdown("##### 📌 Workflow Milestones & Unified Direct Actions")
m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)

with m_c1:
    st.markdown(
        """
        <div class="milestone-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span class="neo-badge badge-purple" style="font-size: 0.62rem;">Step 1</span>
                <span class="neo-badge badge-dark" style="font-size: 0.62rem;">INGEST</span>
            </div>
            <div style="font-weight: 900; font-size: 0.92rem; margin-bottom: 4px;">📁 Load Dataset</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">CSV, Excel, Vision & Split</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Ingest ↗", key="m_btn_1", use_container_width=True):
        st.session_state.active_module = "ingest"
        st.rerun()

with m_c2:
    st.markdown(
        """
        <div class="milestone-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span class="neo-badge badge-yellow" style="font-size: 0.62rem;">Step 2</span>
                <span class="neo-badge badge-dark" style="font-size: 0.62rem;">CLEAN</span>
            </div>
            <div style="font-weight: 900; font-size: 0.92rem; margin-bottom: 4px;">🧹 Auto Cleaner</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">Types, Impute, Outliers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Cleaner ↗", key="m_btn_2", use_container_width=True):
        st.session_state.active_module = "clean"
        st.rerun()

with m_c3:
    st.markdown(
        """
        <div class="milestone-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span class="neo-badge badge-cyan" style="font-size: 0.62rem;">Step 3</span>
                <span class="neo-badge badge-dark" style="font-size: 0.62rem;">POWER BI</span>
            </div>
            <div style="font-weight: 900; font-size: 0.92rem; margin-bottom: 4px;">📊 Power BI</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">Slicers, Donut, Matrix</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Power BI ↗", key="m_btn_3", use_container_width=True):
        st.session_state.active_module = "eda"
        st.rerun()

with m_c4:
    st.markdown(
        """
        <div class="milestone-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span class="neo-badge badge-lime" style="font-size: 0.62rem;">Step 4</span>
                <span class="neo-badge badge-dark" style="font-size: 0.62rem;">AUTOML</span>
            </div>
            <div style="font-weight: 900; font-size: 0.92rem; margin-bottom: 4px;">🤖 AutoML Studio</div>
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 600;">3+ Models & Predict</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open AutoML ↗", key="m_btn_4", use_container_width=True):
        st.session_state.active_module = "automl"
        st.rerun()

with m_c5:
    st.markdown(
        """
        <div class="milestone-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span class="neo-badge badge-purple" style="font-size: 0.62rem;">Step 5</span>
                <span class="neo-badge badge-dark" style="font-size: 0.62rem;">AI INTEL</span>
            </div>
            <div style="font-weight: 900; font-size: 0.86rem; margin-bottom: 4px;">💬 Instant Voice & Text Dataset Intelligence</div>
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 600;">Mic STT & Multilingual Text AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Intelligence ↗", key="m_btn_5", use_container_width=True):
        st.session_state.active_module = "chat"
        st.rerun()

with m_c6:
    st.markdown(
        """
        <div class="milestone-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span class="neo-badge badge-yellow" style="font-size: 0.62rem;">Step 6</span>
                <span class="neo-badge badge-dark" style="font-size: 0.62rem;">REPORTS</span>
            </div>
            <div style="font-weight: 900; font-size: 0.86rem; margin-bottom: 4px;">📑 Executive Reports</div>
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 600;">Download All Q&A, PDF & Word</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open Reports ↗", key="m_btn_6", use_container_width=True):
        st.session_state.active_module = "report"
        st.rerun()


st.markdown("<hr style='border: 1.5px solid #000; margin-top: 25px; margin-bottom: 25px;'>", unsafe_allow_html=True)


# =============================================================================
# MODULE 1: UNIVERSAL INGESTION & AUTO-SPLIT
# =============================================================================
if st.session_state.active_module == "ingest":
    st.markdown("### 📁 Universal Ingestion & Auto-Split Engine")
    
    ingest_tab1, ingest_tab2 = st.tabs(["📄 Single File Ingestion", "📂 Folder & ZIP Auto-Detect"])

    with ingest_tab1:
        uploaded_file = st.file_uploader(
            "Upload any dataset (CSV, TSV, XLSX, XLS, JSON, Parquet):",
            type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
            help="Encoding and delimiter will be detected automatically.",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            ext = os.path.splitext(filename)[1].lower()

            selected_sheet = None
            if ext in [".xlsx", ".xls"]:
                from src.loader import get_excel_sheets
                sheets = get_excel_sheets(file_bytes, ext)
                if len(sheets) > 1:
                    selected_sheet = st.selectbox("Workbook contains multiple sheets. Choose sheet:", sheets)

            max_sample_rows = None
            if len(file_bytes) > 50 * 1024 * 1024:
                st.warning("⚠️ File exceeds 50MB. Downsampling enabled.")
                if st.checkbox("Enable sampling (read first 50,000 rows)", value=True):
                    max_sample_rows = 50000

            df, meta = load_tabular_from_bytes(file_bytes, filename, selected_sheet, max_sample_rows)

            if meta.get("error"):
                st.error(f"❌ {meta['error']}")
            else:
                save_raw_file(file_bytes, filename)
                st.session_state.raw_df = df
                st.session_state.cleaned_df = df.copy()
                st.session_state.raw_filename = filename
                st.session_state.dataset_type = "tabular"
                st.session_state.tabular_splits = None
                st.session_state.ml_results = None

                st.success(f"✅ Successfully ingested **{filename}** (Saved untouched in `data/raw/`).")

    with ingest_tab2:
        folder_mode = st.radio("Provide Input Via:", ["🗂️ Local Folder Path", "📦 Upload ZIP Archive"], horizontal=True)
        target_scan_dir = None

        if folder_mode == "🗂️ Local Folder Path":
            folder_input_path = st.text_input("Enter absolute or relative folder path:", value=SAMPLE_PATHS["image_folder"])
            if st.button("🔍 Scan Directory", use_container_width=False):
                if not os.path.exists(folder_input_path):
                    st.error(f"❌ Directory not found: `{folder_input_path}`")
                else:
                    target_scan_dir = folder_input_path
        else:
            uploaded_zip = st.file_uploader("Upload ZIP dataset archive:", type=["zip"])
            if uploaded_zip is not None:
                if st.button("📦 Extract & Scan ZIP", use_container_width=False):
                    zip_bytes = uploaded_zip.getvalue()
                    extracted_path, _ = extract_zip_to_raw(zip_bytes, uploaded_zip.name)
                    target_scan_dir = extracted_path

        if target_scan_dir:
            scan_res = scan_dataset_directory(target_scan_dir)
            st.markdown(f"**Scan Result**: {scan_res['summary']}")

            if scan_res["mode"] == "images_per_class":
                st.info(f"🖼️ Detected **Image Classification Dataset** with **{len(scan_res['classes'])} classes**.")
                st.dataframe(pd.DataFrame(list(scan_res["classes"].items()), columns=["Class Label", "Image Count"]), use_container_width=True)

                if st.button("🚀 Confirm and Ingest Image Dataset"):
                    items = [(p, os.path.basename(os.path.dirname(p))) for p in scan_res["images"]]
                    st.session_state.raw_df = None
                    st.session_state.cleaned_df = None
                    st.session_state.image_data_items = items
                    st.session_state.image_classes = scan_res["classes"]
                    st.session_state.raw_filename = os.path.basename(target_scan_dir)
                    st.session_state.dataset_type = "image"
                    st.session_state.image_splits = None
                    st.session_state.image_model_results = None
                    st.success("✅ Image dataset successfully registered!")
                    st.rerun()

            elif scan_res["mode"] in ["tables_mergeable", "tables_single"]:
                if scan_res.get("can_append_tables"):
                    st.info(f"📊 Multiple tables with identical columns found ({len(scan_res['tables'])} files).")
                    if st.button("🔗 Append & Combine All Tables"):
                        combined_df, _ = load_and_append_tables(scan_res["tables"])
                        if combined_df is not None:
                            st.session_state.raw_df = combined_df
                            st.session_state.cleaned_df = combined_df.copy()
                            st.session_state.raw_filename = f"Combined_{len(scan_res['tables'])}_Tables"
                            st.session_state.dataset_type = "tabular"
                            st.rerun()

    # Data Profiling & Split
    if st.session_state.dataset_type == "tabular" and st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        summary = compute_dataset_summary(df)

        st.markdown("---")
        st.subheader("📊 Dataset Health & Dimensions")

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Total Records</div><div class="neo-card-val">{summary['rows']:,}</div></div>""", unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Features</div><div class="neo-card-val">{summary['columns']}</div></div>""", unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Missing Cells</div><div class="neo-card-val">{summary['missing_pct']}%</div></div>""", unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Duplicates</div><div class="neo-card-val">{summary['duplicate_pct']}%</div></div>""", unsafe_allow_html=True)
        with kpi5:
            st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Memory Size</div><div class="neo-card-val">{summary['memory_mb']} MB</div></div>""", unsafe_allow_html=True)

        st.markdown("##### 🔍 Data Preview (First 10 Records)")
        st.dataframe(df.head(10), use_container_width=True)

        # Tabular Stratified Split
        st.markdown("---")
        st.subheader("✂️ Stratified Split & Zero Data Leakage Guard")
        
        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            target_col = st.selectbox("Select Target / Dependent Column for Split:", df.columns)
        with sp2:
            train_slider = st.slider("Train Split (%)", 50, 85, 70, step=5)
        with sp3:
            val_slider = st.slider("Validation Split (%)", 5, 25, 15, step=5)
        
        test_pct_val = 100 - (train_slider + val_slider)
        st.caption(f"Calculated Hold-out Test Split: **{test_pct_val}%**")

        if st.button("⚡ Execute Stratified Split", use_container_width=True):
            train_df, val_df, test_df, s_report = split_tabular_data(
                df,
                target_col=target_col,
                train_ratio=train_slider / 100.0,
                val_ratio=val_slider / 100.0,
                test_ratio=test_pct_val / 100.0,
                random_state=42,
            )
            st.session_state.tabular_splits = (train_df, val_df, test_df, s_report)
            st.session_state.ml_target = target_col
            st.success("✅ Dataset split completed successfully!")

        if st.session_state.tabular_splits is not None:
            _, _, _, rep = st.session_state.tabular_splits
            if rep.get("leakage_warning"):
                st.error(f"🛑 {rep['leakage_warning']}")
            else:
                st.info("🛡️ Zero Data Leakage Verified: Checksum MD5 validation passed across train and test partitions.")
            st.plotly_chart(create_split_distribution_chart(rep), use_container_width=True)


# =============================================================================
# MODULE 2: AUTO DATA CLEANING
# =============================================================================
elif st.session_state.active_module == "clean":
    st.markdown("### 🧹 Autonomous Data Cleaning Engine")

    if st.session_state.raw_df is None:
        st.warning("Please upload a tabular dataset or load a sample first.")
    else:
        df = st.session_state.raw_df
        st.markdown("Configure your autonomous cleaning strategies below:")

        c1, c2, c3 = st.columns(3)
        with c1:
            num_strategy = st.selectbox("Numeric Missing Strategy:", ["median", "mean", "mode", "constant", "drop", "ffill"], index=0)
        with c2:
            cat_strategy = st.selectbox("Categorical Missing Strategy:", ["mode", "constant", "drop", "ffill"], index=0)
        with c3:
            outlier_strategy = st.selectbox("Outlier Handling (IQR Bounds):", ["clip", "drop", "ignore"], index=0)

        c4, c5 = st.columns(2)
        with c4:
            clean_headers = st.checkbox("Standardize Column Names (snake_case)", value=True)
        with c5:
            auto_coerce = st.checkbox("Auto-Coerce Data Types (Numbers/Dates as text)", value=True)

        if st.button("🚀 Run Full Autonomous Cleaning Pipeline", use_container_width=True):
            with st.spinner("Executing transformations and generating audit log..."):
                cleaned_df, log = execute_full_cleaning_pipeline(
                    df,
                    standardize_headers=clean_headers,
                    coerce_types=auto_coerce,
                    numeric_impute_strategy=num_strategy,
                    cat_impute_strategy=cat_strategy,
                    outlier_method=outlier_strategy,
                )
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_log = log
            st.success("✅ Autonomous cleaning executed successfully!")

        if st.session_state.cleaned_df is not None:
            c_df = st.session_state.cleaned_df
            st.markdown("---")
            st.markdown("### 📊 Raw vs. Cleaned Dataset Comparison")

            raw_summary = compute_dataset_summary(df)
            clean_summary = compute_dataset_summary(c_df)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Missing Cells Before</div><div class="neo-card-val" style="color: #ef4444;">{raw_summary['missing_cells']:,}</div></div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Missing Cells After</div><div class="neo-card-val" style="color: #22c55e;">{clean_summary['missing_cells']:,}</div></div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Duplicates Purged</div><div class="neo-card-val">{raw_summary['duplicate_rows'] - clean_summary['duplicate_rows']:,}</div></div>""", unsafe_allow_html=True)
            with k4:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Clean Columns</div><div class="neo-card-val">{clean_summary['columns']}</div></div>""", unsafe_allow_html=True)

            st.markdown("##### 📜 Detailed Audit Cleaning Log")
            st.text_area("Audit History", value="\n".join(st.session_state.cleaning_log), height=180)

            st.markdown("##### 📥 Export Cleaned Dataset")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                csv_bytes = get_downloadable_bytes(c_df, "csv")
                st.download_button("📥 Download Cleaned CSV", data=csv_bytes, file_name="cleaned_dataset.csv", mime="text/csv", use_container_width=True)
            with col_d2:
                excel_bytes = get_downloadable_bytes(c_df, "excel")
                st.download_button("📥 Download Cleaned Excel (.xlsx)", data=excel_bytes, file_name="cleaned_dataset.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            with col_d3:
                log_txt = "\n".join(st.session_state.cleaning_log).encode("utf-8")
                st.download_button("📥 Download Audit Log (.txt)", data=log_txt, file_name="cleaning_audit_log.txt", mime="text/plain", use_container_width=True)


# =============================================================================
# MODULE 3: POWER BI EXECUTIVE DASHBOARD
# =============================================================================
elif st.session_state.active_module == "eda":
    st.markdown("### 📊 Power BI Executive Analytics Dashboard")
    st.caption("Interactive Slicers Ribbon, KPI Multi-Row Scorecards, Category Donut, Clustered Bar, Treemap, KPI Gauge & Matrix")

    active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df
    if active_df is None:
        st.warning("Please upload a dataset or select a sample first.")
    else:
        cat_cols = list(active_df.select_dtypes(exclude=["number"]).columns)
        num_cols = list(active_df.select_dtypes(include=["number"]).columns)

        # 1. POWER BI INTERACTIVE SLICERS RIBBON
        with st.expander("🔍 Power BI Slicers & Dynamic Cross-Filtering Ribbon", expanded=True):
            s_c1, s_c2, s_c3, s_c4 = st.columns([1.5, 1.5, 1.8, 1.2])
            filtered_df = active_df.copy()

            with s_c1:
                if cat_cols:
                    slicer_cat1 = st.selectbox("📌 Primary Segment Slicer:", cat_cols, key="pbi_slicer_cat1")
                    unique_vals1 = list(active_df[slicer_cat1].dropna().unique())
                    chosen_vals1 = st.multiselect(
                        f"Filter {slicer_cat1}:",
                        options=unique_vals1,
                        default=unique_vals1[:min(len(unique_vals1), 5)],
                        key="pbi_slicer_vals1",
                    )
                    if chosen_vals1:
                        filtered_df = filtered_df[filtered_df[slicer_cat1].isin(chosen_vals1)]

            with s_c2:
                if len(cat_cols) > 1:
                    slicer_cat2 = st.selectbox("📌 Secondary Segment Slicer:", cat_cols, index=1, key="pbi_slicer_cat2")
                    unique_vals2 = list(active_df[slicer_cat2].dropna().unique())
                    chosen_vals2 = st.multiselect(
                        f"Filter {slicer_cat2}:",
                        options=unique_vals2,
                        default=unique_vals2,
                        key="pbi_slicer_vals2",
                    )
                    if chosen_vals2:
                        filtered_df = filtered_df[filtered_df[slicer_cat2].isin(chosen_vals2)]
                elif num_cols:
                    st.caption("Only 1 categorical column detected in dataset.")

            with s_c3:
                if num_cols:
                    slicer_num = st.selectbox("📏 Numeric Range Slicer:", num_cols, key="pbi_slicer_num")
                    min_val = float(active_df[slicer_num].min())
                    max_val = float(active_df[slicer_num].max())
                    if min_val < max_val:
                        selected_range = st.slider(
                            f"Filter Range for {slicer_num}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key="pbi_slider_num",
                        )
                        filtered_df = filtered_df[
                            (filtered_df[slicer_num] >= selected_range[0]) & (filtered_df[slicer_num] <= selected_range[1])
                        ]

            with s_c4:
                search_q = st.text_input("🔎 Search Text Slicer:", placeholder="Filter keyword...", key="pbi_search_box")
                if search_q:
                    mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_q, case=False).any(), axis=1)
                    filtered_df = filtered_df[mask]

                if st.button("🔄 Reset Slicers", key="reset_slicers_btn", use_container_width=True):
                    filtered_df = active_df.copy()
                    st.rerun()

            filter_pct = (len(filtered_df) / max(len(active_df), 1)) * 100
            st.markdown(
                f"""
                <div style="background: #0078D4; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: 800; font-size: 0.82rem; margin-top: 8px; display: inline-flex; align-items: center; gap: 8px;">
                    <span>⚡ Power BI Cross-Filtering Active:</span>
                    <span>Showing <b>{len(filtered_df):,}</b> of <b>{len(active_df):,}</b> records (<b>{filter_pct:.1f}%</b> of total)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 2. POWER BI KPI SCORECARD ROW (5-CARD MULTI-ROW)
        st.markdown("##### 📌 Power BI Executive KPI Cards")
        k_c1, k_c2, k_c3, k_c4, k_c5 = st.columns(5)

        with k_c1:
            st.markdown(
                f"""
                <div class="pbi-scorecard">
                    <div class="pbi-scorecard-label">Active Volume</div>
                    <div class="pbi-scorecard-num">{len(filtered_df):,}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k_c2:
            if num_cols:
                p_metric = num_cols[0]
                total_sum = filtered_df[p_metric].sum() if not filtered_df.empty else 0
                sum_formatted = f"{total_sum:,.1f}" if abs(total_sum) < 1000000 else f"{total_sum/1000000:.2f}M"
                st.markdown(
                    f"""
                    <div class="pbi-scorecard">
                        <div class="pbi-scorecard-label">Total {p_metric}</div>
                        <div class="pbi-scorecard-num" style="color: #0078D4;">{sum_formatted}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="pbi-scorecard">
                        <div class="pbi-scorecard-label">Features</div>
                        <div class="pbi-scorecard-num">{len(filtered_df.columns)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with k_c3:
            if num_cols:
                p_metric = num_cols[0]
                avg_val = round(filtered_df[p_metric].mean(), 2) if not filtered_df.empty else 0
                st.markdown(
                    f"""
                    <div class="pbi-scorecard">
                        <div class="pbi-scorecard-label">Average {p_metric}</div>
                        <div class="pbi-scorecard-num" style="color: #107C41;">{avg_val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="pbi-scorecard">
                        <div class="pbi-scorecard-label">Data Types</div>
                        <div class="pbi-scorecard-num">{filtered_df.dtypes.nunique()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with k_c4:
            p_cat = cat_cols[0] if cat_cols else "Columns"
            distinct_cnt = filtered_df[p_cat].nunique() if cat_cols else len(filtered_df.columns)
            st.markdown(
                f"""
                <div class="pbi-scorecard">
                    <div class="pbi-scorecard-label">Distinct {p_cat}</div>
                    <div class="pbi-scorecard-num" style="color: #5C2D91;">{distinct_cnt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k_c5:
            completeness = round(100 - (filtered_df.isnull().sum().sum() / max(filtered_df.size, 1) * 100), 1)
            comp_color = "#107C41" if completeness >= 95 else "#D83B01"
            st.markdown(
                f"""
                <div class="pbi-scorecard">
                    <div class="pbi-scorecard-label">Data Quality</div>
                    <div class="pbi-scorecard-num" style="color: {comp_color};">{completeness}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. POWER BI VISUAL DECOMPOSITION GRID
        st.markdown("---")
        st.markdown("##### 📈 Power BI Visual Decomposition Grid")

        v_row1_col1, v_row1_col2 = st.columns(2)
        with v_row1_col1:
            if cat_cols:
                donut_target = cat_cols[0]
                fig_donut, ins_donut = generate_powerbi_donut(filtered_df, donut_target)
                st.plotly_chart(fig_donut, use_container_width=True)
                st.info(ins_donut)
            else:
                st.caption("No categorical feature available for Donut visual.")

        with v_row1_col2:
            if cat_cols:
                bar_cat = cat_cols[min(len(cat_cols)-1, 1)]
                bar_num = num_cols[0] if num_cols else None
                fig_bar, ins_bar = generate_powerbi_clustered_bar(filtered_df, bar_cat, bar_num)
                st.plotly_chart(fig_bar, use_container_width=True)
                st.info(ins_bar)
            else:
                st.caption("No categorical feature available for Clustered Bar visual.")

        v_row2_col1, v_row2_col2 = st.columns(2)
        with v_row2_col1:
            fig_corr, ins_corr = generate_correlation_heatmap(filtered_df)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
                st.info(ins_corr)
            else:
                st.caption(ins_corr)

        with v_row2_col2:
            date_candidates = [c for c in filtered_df.columns if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(filtered_df[c])]
            if date_candidates and num_cols:
                fig_area, ins_area = generate_powerbi_area(filtered_df, date_candidates[0], num_cols[0])
                if fig_area:
                    st.plotly_chart(fig_area, use_container_width=True)
                    st.info(ins_area)
            elif num_cols:
                fig_dist, ins_dist = generate_distribution_chart(filtered_df, num_cols[0])
                st.plotly_chart(fig_dist, use_container_width=True)
                st.info(ins_dist)

        # 4. POWER BI TREEMAP & GAUGE ROW
        v_row3_col1, v_row3_col2 = st.columns(2)
        with v_row3_col1:
            if cat_cols:
                st.markdown("##### 🗺️ Power BI Treemap Hierarchy")
                tm_col = cat_cols[0]
                tm_val = num_cols[0] if num_cols else None
                fig_tm, ins_tm = generate_powerbi_treemap(filtered_df, tm_col, tm_val)
                st.plotly_chart(fig_tm, use_container_width=True)
                st.info(ins_tm)
        with v_row3_col2:
            if num_cols:
                st.markdown("##### ⏱️ Power BI Target Gauge")
                target_col_gauge = num_cols[0]
                cur_mean = float(filtered_df[target_col_gauge].mean()) if not filtered_df.empty else 0
                max_metric = float(active_df[target_col_gauge].max()) if not active_df.empty else 100
                fig_gauge = generate_powerbi_gauge(cur_mean, 0, max_metric, f"Average {target_col_gauge}")
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption(f"Gauge tracking average {target_col_gauge} across active filters.")

        # 5. POWER BI MATRIX / PIVOT SUMMARY TABLE
        if cat_cols and num_cols:
            st.markdown("---")
            st.markdown("##### 📋 Power BI Executive Matrix Visual")
            st.caption("Drill-down aggregation table: Count, Sum, Mean, Min, Max, and Segment Share %")
            m_cat = st.selectbox("Matrix Row Dimension (Category):", cat_cols, key="matrix_cat_sel")
            m_num = st.selectbox("Matrix Value Dimension (Metric):", num_cols, key="matrix_num_sel")
            matrix_df = generate_powerbi_matrix(filtered_df, m_cat, m_num)
            st.dataframe(matrix_df, use_container_width=True)

        # 6. POWER BI CUSTOM VISUAL STUDIO
        st.markdown("---")
        st.markdown("##### 🛠️ Power BI Custom Visual Studio")
        pbi_b1, pbi_b2, pbi_b3, pbi_b4 = st.columns(4)
        with pbi_b1:
            pbi_vtype = st.selectbox("Visual Type:", ["Donut", "Clustered Bar", "Treemap", "Line", "Scatter", "Box", "Histogram"], key="pbi_studio_type")
        with pbi_b2:
            pbi_x = st.selectbox("X-Axis / Category:", filtered_df.columns, key="pbi_studio_x")
        with pbi_b3:
            pbi_y = st.selectbox("Y-Axis / Metric:", ["None"] + list(filtered_df.columns), key="pbi_studio_y")
        with pbi_b4:
            pbi_grp = st.selectbox("Color / Legend:", ["None"] + list(filtered_df.columns), key="pbi_studio_grp")

        custom_pbi_fig = build_custom_chart(filtered_df, pbi_vtype, pbi_x, pbi_y, pbi_grp)
        st.plotly_chart(custom_pbi_fig, use_container_width=True)



# =============================================================================
# MODULE 4: AUTOML STUDIO (TRAIN & PREDICT)
# =============================================================================
elif st.session_state.active_module == "automl":
    st.markdown("### 🤖 Autonomous Machine Learning Engine")

    if st.session_state.dataset_type == "tabular":
        active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df
        if active_df is None:
            st.warning("Please upload a dataset or select a sample first.")
        else:
            available_cols = list(active_df.columns)
            target_idx = 0
            if st.session_state.ml_target in available_cols:
                target_idx = available_cols.index(st.session_state.ml_target)
            target_col = st.selectbox("Select Target Feature to Predict:", available_cols, index=target_idx, key="automl_target_select")
            st.session_state.ml_target = target_col
            task_type = infer_task_type(active_df[target_col])
            st.markdown(f"**Inferred Task Type**: `<span class='neo-badge badge-lime'>{task_type.upper()}</span>`", unsafe_allow_html=True)

            if st.session_state.tabular_splits is None:
                train_df, val_df, test_df, s_rep = split_tabular_data(active_df, target_col=target_col)
                st.session_state.tabular_splits = (train_df, val_df, test_df, s_rep)

            train_df, _, test_df, _ = st.session_state.tabular_splits

            if st.button("🚀 Train 3+ ML Pipelines (1-Click)", use_container_width=True):
                with st.spinner("Executing scikit-learn pipelines with cross-validation..."):
                    res = train_tabular_models(train_df, test_df, target_col=target_col)
                    st.session_state.ml_results = res
                st.success("✅ Machine Learning training completed!")

            if st.session_state.ml_results:
                res = st.session_state.ml_results
                st.markdown("### 🏆 Model Comparison Leaderboard (Evaluated on Held-out Test Set)")
                st.dataframe(res["leaderboard"], use_container_width=True)
                st.markdown(f"**Best Selected Pipeline**: `{res['best_model_name']}`")

                cA, cB = st.columns(2)
                with cA:
                    if res.get("confusion_matrix_fig"):
                        st.plotly_chart(res["confusion_matrix_fig"], use_container_width=True)
                with cB:
                    if res.get("feature_importance_fig"):
                        st.plotly_chart(res["feature_importance_fig"], use_container_width=True)

                # Dynamic Prediction Form
                st.markdown("---")
                st.subheader(f"🔮 Dynamic Inference Form ({res['best_model_name']})")
                with st.form("tabular_prediction_form"):
                    input_data = {}
                    cols_to_input = res["feature_cols"]
                    form_cols = st.columns(min(len(cols_to_input), 3))
                    for idx, c in enumerate(cols_to_input):
                        with form_cols[idx % 3]:
                            if pd.api.types.is_numeric_dtype(active_df[c]):
                                med = float(active_df[c].median())
                                input_data[c] = st.number_input(f"{c}", value=med)
                            else:
                                opts = list(active_df[c].dropna().unique())
                                input_data[c] = st.selectbox(f"{c}", opts)
                    
                    if st.form_submit_button("⚡ Predict"):
                        pred_val = predict_tabular_sample(res["best_pipeline"], input_data)
                        st.markdown(
                            f"""
                            <div style="background: #bef264; border: 2.5px solid #000; border-radius: 16px; box-shadow: 4px 4px 0px #000; padding: 20px; margin-top: 10px;">
                                <div style="color: #000; font-size: 0.9rem; font-weight: 800; text-transform: uppercase;">Prediction Output:</div>
                                <div style="color: #000; font-size: 2.2rem; font-weight: 900;">{pred_val}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    elif st.session_state.dataset_type == "image":
        items = st.session_state.image_data_items
        st.markdown("### 📷 MobileNetV2 Deep Transfer Learning Studio")

        if st.session_state.image_splits is None:
            train_l, val_l, test_l, rep = split_image_dataset(items)
            st.session_state.image_splits = (train_l, val_l, test_l, rep)

        train_l, val_l, test_l, _ = st.session_state.image_splits

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            epochs = st.number_input("Epochs", min_value=1, max_value=20, value=3)
        with col2:
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
        with col3:
            img_size = st.selectbox("Image Size", [128, 224], index=0)
        with col4:
            quick_test = st.checkbox("Quick Test Mode", value=True)

        if st.button("🚀 Launch MobileNetV2 Transfer Learning", use_container_width=True):
            p_bar = st.progress(0.0)
            status_t = st.empty()

            def cb(cur, tot, m):
                p_bar.progress(cur / tot)
                status_t.markdown(f"**Epoch {cur}/{tot}** — Loss: `{m['train_loss']:.4f}` | Acc: `{m['train_acc']:.1f}%`")

            with st.spinner("Training MobileNetV2..."):
                img_res = train_mobilenet_v2(
                    train_l, val_l, test_l,
                    epochs=epochs, batch_size=batch_size, img_size=img_size,
                    quick_test_mode=quick_test, progress_callback=cb,
                )
                st.session_state.image_model_results = img_res
            st.success("✅ Vision model training completed!")

        if st.session_state.image_model_results:
            res = st.session_state.image_model_results
            meta = res["metadata"]
            m_metrics = meta["metrics"]

            st.markdown("#### 🏆 Test Set Performance")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Test Accuracy</div><div class="neo-card-val">{m_metrics['test_accuracy']}%</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Weighted F1</div><div class="neo-card-val">{m_metrics['test_f1']}%</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Precision</div><div class="neo-card-val">{m_metrics['test_precision']}%</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="neo-card"><div class="neo-card-title">Recall</div><div class="neo-card-val">{m_metrics['test_recall']}%</div></div>""", unsafe_allow_html=True)

            c_curve, c_mat = st.columns(2)
            with c_curve:
                st.plotly_chart(create_training_curves_chart(res["history"]), use_container_width=True)
            with c_mat:
                st.plotly_chart(create_confusion_matrix_chart(meta["confusion_matrix"], meta["classes"]), use_container_width=True)


# =============================================================================
# MODULE 5: CHAT WITH DATA (VOICE MIC INPUT + TAMIL + GROQ)
# =============================================================================
elif st.session_state.active_module == "chat":
    st.markdown("### 💬 Ask Your Data — Multilingual Voice & AI Chatbot")
    st.caption("Voice Microphone Speech-to-Text via Groq Whisper. Type or speak any dataset inquiry.")

    active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df
    if active_df is None:
        st.warning("Please upload a dataset or select a sample first.")
    else:
        # Settings Bar for Chatbot
        c_mod1, c_mod2, c_mod3 = st.columns([1.5, 1.5, 1])
        with c_mod1:
            st.session_state.groq_api_key = st.text_input("Groq API Key (Optional):", value=st.session_state.groq_api_key, type="password", placeholder="gsk_...")
        with c_mod2:
            st.session_state.selected_groq_model = st.selectbox("Groq Model:", SUPPORTED_GROQ_MODELS, index=0)
        with c_mod3:
            st.session_state.language = st.selectbox("Language / மொழி:", ["English", "தமிழ் (Tamil)"], index=0 if st.session_state.language == "English" else 1)

        # Voice Input Mic Row (Only for asking questions)
        voice_c1, voice_c2 = st.columns([1.2, 2.8])
        with voice_c1:
            st.markdown("##### 🎙️ Voice Input (Mic)")
            st.caption("Voice is strictly for speaking questions via microphone STT (Groq Whisper).")
            if MIC_AVAILABLE:
                audio_record = mic_recorder(
                    start_prompt="🔴 Speak Question (Mic)",
                    stop_prompt="⏹️ Transcribe Voice",
                    key="voice_mic_q",
                )
                if audio_record and "bytes" in audio_record:
                    st.info("Transcribing spoken question via Groq Whisper...")
                    transcription, err = transcribe_audio_groq(
                        audio_record["bytes"],
                        st.session_state.groq_api_key,
                        model=st.session_state.selected_whisper_model,
                    )
                    if transcription:
                        st.session_state.voice_prompt = transcription
                        st.success(f"Recognized Spoken Query: *\"{transcription}\"*")
                    elif err:
                        st.warning(err)
            else:
                st.caption("Microphone recorder module active.")

        with voice_c2:
            st.markdown("##### 📥 Instant Download All Responses")
            d_c1, d_c2, d_c3 = st.columns(3)
            with d_c1:
                chat_json_str = json.dumps([
                    {
                        "role": m.get("role"),
                        "content": str(m.get("content", "")),
                        "code": m.get("code"),
                        "timestamp": m.get("timestamp", ""),
                    } for m in st.session_state.chat_history
                ], indent=2)
                st.download_button(
                    "📥 Download All Q&A (JSON)",
                    data=chat_json_str.encode("utf-8"),
                    file_name=f"Chat_Responses_{st.session_state.raw_filename or 'Dataset'}.json",
                    mime="application/json",
                    disabled=len(st.session_state.chat_history) == 0,
                    use_container_width=True,
                )
            with d_c2:
                transcript_lines = []
                for idx, m in enumerate(st.session_state.chat_history, start=1):
                    role_str = "USER QUESTION" if m.get("role") == "user" else "AI ANALYST RESPONSE"
                    t_stamp = f" [{m.get('timestamp')}]" if m.get("timestamp") else ""
                    transcript_lines.append(f"#{idx} {role_str}{t_stamp}:\n{m.get('content')}")
                    if m.get("code"):
                        transcript_lines.append(f"Executed Code:\n{m.get('code')}")
                transcript_txt = "\n\n" + ("=" * 45) + "\n\n".join(transcript_lines)
                st.download_button(
                    "📥 Download Transcript (TXT)",
                    data=transcript_txt.encode("utf-8"),
                    file_name=f"Chat_Transcript_{st.session_state.raw_filename or 'Dataset'}.txt",
                    mime="text/plain",
                    disabled=len(st.session_state.chat_history) == 0,
                    use_container_width=True,
                )
            with d_c3:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

        # Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "code" in msg and msg["code"]:
                    with st.expander("🛠️ View Generated Python Code"):
                        st.code(msg["code"], language="python")
                if "result_df" in msg and msg["result_df"] is not None:
                    st.dataframe(msg["result_df"], use_container_width=True)

        # Chat Input Box
        prompt = st.chat_input("Ask any question in English or தமிழ் (e.g., 'What is the average fare?' or 'எத்தனை பதிவுகள் உள்ளன?'):")
        
        if hasattr(st.session_state, "voice_prompt") and st.session_state.voice_prompt:
            prompt = st.session_state.voice_prompt
            st.session_state.voice_prompt = None

        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M:%S")})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("AI analyzing question & generating query execution..."):
                    if st.session_state.groq_api_key:
                        llm_res = generate_groq_pandas_code(
                            prompt,
                            active_df,
                            api_key=st.session_state.groq_api_key,
                            model_name=st.session_state.selected_groq_model,
                            language=st.session_state.language,
                        )
                        if llm_res.get("error"):
                            st.caption(f"Note: {llm_res['error']}. Using local analytical engine.")
                            ans_pack = offline_fallback_analyst(prompt, active_df, st.session_state.language)
                            gen_code = ans_pack["code"]
                            exec_res = {"result": ans_pack["result"], "type": ans_pack["type"], "error": None}
                            answer_text = ans_pack["answer"]
                        else:
                            gen_code = llm_res["code"]
                            exec_res = execute_safe_query(gen_code, active_df)
                            answer_text = f"Result of `{prompt}` using `{llm_res['model_used']}`:"
                    else:
                        ans_pack = offline_fallback_analyst(prompt, active_df, st.session_state.language)
                        gen_code = ans_pack["code"]
                        exec_res = {"result": ans_pack["result"], "type": ans_pack["type"], "error": None}
                        answer_text = ans_pack["answer"]

                    st.markdown(answer_text)
                    res_obj = exec_res["result"]
                    saved_df = None

                    if exec_res["type"] == "dataframe":
                        st.dataframe(res_obj, use_container_width=True)
                        saved_df = res_obj
                    elif exec_res["type"] == "chart":
                        st.plotly_chart(res_obj, use_container_width=True)
                    elif exec_res["type"] == "scalar":
                        st.markdown(f"**Computed Value**: `{res_obj}`")
                    elif exec_res["type"] == "error":
                        st.error(exec_res["error"])

                    with st.expander("🛠️ View Generated Python Code"):
                        st.code(gen_code, language="python")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "code": gen_code,
                        "result_df": saved_df,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                    })


# =============================================================================
# MODULE 6: AUTOMATED EXECUTIVE REPORT GENERATOR (ALL RESPONSES DOWNLOAD)
# =============================================================================
elif st.session_state.active_module == "report":
    st.markdown("### 📑 Automated Executive Analytical Report Generator")
    st.caption("Download comprehensive session reports containing all dataset KPIs, auto-cleaning audit logs, AutoML model metrics, and ALL Chatbot Q&A responses.")

    active_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df
    if active_df is None:
        st.warning("Please upload a dataset or select a sample first.")
    else:
        summary = compute_dataset_summary(active_df)

        # Overview Card of Active Session Contents
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 2.5px solid #000; border-radius: 16px; box-shadow: 4px 4px 0px #000; padding: 18px; margin-bottom: 22px;">
                <div style="font-size: 1.1rem; font-weight: 900; margin-bottom: 6px;">
                    ✦ Session Report Package: <b>{st.session_state.raw_filename or 'Dataset'}</b>
                </div>
                <div style="font-size: 0.88rem; color: #475569; font-weight: 600;">
                    Ready for 1-click download. Every single response, audit log entry, and model benchmark is included below:
                </div>
                <div style="display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap;">
                    <span class="neo-badge badge-purple">💬 {len(st.session_state.chat_history)} Chat Inquiries Logged</span>
                    <span class="neo-badge badge-lime">🧹 {len(st.session_state.cleaning_log)} Cleaning Transformations</span>
                    <span class="neo-badge badge-cyan">🤖 {'AutoML Benchmarks Ready' if st.session_state.ml_results else 'AutoML Not Run'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rep_c1, rep_c2, rep_c3 = st.columns(3)
        with rep_c1:
            st.markdown("#### 📄 Executive PDF Report")
            st.caption("Includes KPIs, full cleaning history, ML benchmarks, and all chatbot Q&A.")
            pdf_bytes = generate_pdf_report(
                dataset_name=st.session_state.raw_filename or "Dataset",
                summary_metrics=summary,
                cleaning_log=st.session_state.cleaning_log,
                ml_results=st.session_state.ml_results,
                chat_history=st.session_state.chat_history,
            )
            st.download_button(
                "📥 Download Complete PDF",
                data=pdf_bytes,
                file_name=f"Executive_Audit_{st.session_state.raw_filename or 'Dataset'}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with rep_c2:
            st.markdown("#### 📝 Microsoft Word (.docx)")
            st.caption("Editable Word document containing all tabular metrics and responses.")
            docx_bytes = generate_word_report(
                dataset_name=st.session_state.raw_filename or "Dataset",
                summary_metrics=summary,
                cleaning_log=st.session_state.cleaning_log,
                ml_results=st.session_state.ml_results,
                chat_history=st.session_state.chat_history,
            )
            st.download_button(
                "📥 Download Word (.docx)",
                data=docx_bytes,
                file_name=f"Executive_Report_{st.session_state.raw_filename or 'Dataset'}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

        with rep_c3:
            st.markdown("#### 📦 Full Session Export (JSON)")
            st.caption("Raw machine-readable export with every single query, result, and metric.")
            json_str = generate_json_session_export(
                dataset_name=st.session_state.raw_filename or "Dataset",
                summary_metrics=summary,
                cleaning_log=st.session_state.cleaning_log,
                ml_results=st.session_state.ml_results,
                chat_history=st.session_state.chat_history,
            )
            st.download_button(
                "📥 Download JSON Bundle",
                data=json_str.encode("utf-8"),
                file_name=f"Session_Responses_{st.session_state.raw_filename or 'Dataset'}.json",
                mime="application/json",
                use_container_width=True,
            )

