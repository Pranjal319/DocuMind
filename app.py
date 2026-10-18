"""DocuMind — Local Document Intelligence Platform.

A modern, minimal AI SaaS interface for local Retrieval-Augmented Generation (RAG).
Built with Streamlit, LangChain, ChromaDB, and local Ollama inference.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import streamlit as st

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    MAX_UPLOAD_SIZE_MB,
    OLLAMA_BASE_URL,
)
from document_processor import (
    DocumentProcessingError,
    EmptyPDFError,
    EncryptedPDFError,
    FileSizeExceededError,
    ScannedPDFError,
    compute_file_hash,
    process_uploaded_pdf,
)
from rag_chain import RAGPipeline, check_ollama_health
from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Page Configuration & Modern SaaS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lightweight, modern dark SaaS styling
st.markdown(
    """
    <style>
    /* Global layout & typography refinements */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4.5rem;
        max-width: 960px;
    }

    /* Brand header */
    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.15rem;
    }
    .brand-spark {
        color: #818CF8;
    }
    .brand-tagline {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-bottom: 0.85rem;
        letter-spacing: -0.01em;
    }

    /* Pill Badges */
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 1.25rem;
    }
    .saas-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background-color: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.22);
        color: #A5B4FC;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    /* Active document status bar */
    .doc-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
        padding: 0.65rem 1rem;
        margin-bottom: 1.2rem;
        color: #E2E8F0;
        font-size: 0.85rem;
    }
    .doc-banner-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #34D399;
        font-weight: 500;
        font-size: 0.76rem;
    }

    /* Citation cards */
    .citation-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 3px solid #6366F1;
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.55rem;
        font-size: 0.83rem;
        line-height: 1.5;
    }
    .citation-meta {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        color: #94A3B8;
        font-size: 0.74rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .citation-tag {
        background: rgba(99, 102, 241, 0.15);
        color: #C7D2FE;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.72rem;
    }

    /* Grounding notice */
    .grounding-alert {
        background: rgba(245, 158, 11, 0.08);
        border-left: 3px solid #F59E0B;
        color: #FCD34D;
        padding: 0.5rem 0.75rem;
        border-radius: 4px;
        font-size: 0.82rem;
        margin: 0.5rem 0;
    }

    /* Sidebar brand header */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 0.15rem;
    }
    .sidebar-sub {
        font-size: 0.75rem;
        color: #64748B;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
def init_session() -> None:
    """Initialize persistent Streamlit variables across user interactions."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_doc" not in st.session_state:
        st.session_state.active_doc = None
    if "vector_manager" not in st.session_state:
        st.session_state.vector_manager = VectorStoreManager()
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(
            vector_store_manager=st.session_state.vector_manager,
            llm_model=DEFAULT_LLM_MODEL,
            temperature=DEFAULT_TEMPERATURE,
        )


init_session()


# -----------------------------------------------------------------------------
# 3. Helper: Render Citation Cards
# -----------------------------------------------------------------------------
def render_citations(citations: List[Dict[str, Any]], expanded: bool = False) -> None:
    """Display expandable source citations beneath an assistant response."""
    if not citations:
        return

    with st.expander(f"📚 Sources & Retrieved Passages ({len(citations)})", expanded=expanded):
        for index, citation in enumerate(citations, start=1):
            source_file = citation.get("source", "document.pdf")
            page_num = citation.get("page", 1)
            total_pages = citation.get("total_pages", "?")
            distance_score = citation.get("score")
            score_text = f"• Distance: {distance_score}" if distance_score is not None else ""

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-meta">
                        <span>Excerpt {index}</span>
                        <span class="citation-tag">📄 {source_file}</span>
                        <span class="citation-tag">Page {page_num} of {total_pages}</span>
                        <span>{score_text}</span>
                    </div>
                    <div style="color: #CBD5E1;">{citation.get('content', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# 4. Sidebar: Document Management & System Health
# -----------------------------------------------------------------------------
with st.sidebar:
    # 4.1 Compact Brand Header
    st.markdown(
        """
        <div class="sidebar-brand">
            <span style="color: #818CF8;">✦</span> DocuMind
        </div>
        <div class="sidebar-sub">Document Intelligence</div>
        """,
        unsafe_allow_html=True,
    )

    # 4.2 Local AI Engine Health Check
    health = check_ollama_health(
        base_url=OLLAMA_BASE_URL,
        llm_model=DEFAULT_LLM_MODEL,
        embed_model=DEFAULT_EMBEDDING_MODEL,
    )

    with st.container(border=True):
        if health["connected"]:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 0.35rem;">
                    <span style="font-size: 0.8rem; font-weight: 600; color: #E2E8F0;">Local Engine</span>
                    <span style="font-size: 0.72rem; color: #34D399; font-weight: 600;">● Connected</span>
                </div>
                <div style="font-size: 0.73rem; color: #94A3B8; line-height: 1.4;">
                    LLM: <code style="color:#C7D2FE;">{DEFAULT_LLM_MODEL}</code><br>
                    Embed: <code style="color:#C7D2FE;">{DEFAULT_EMBEDDING_MODEL}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not (health["has_llm"] and health["has_embedding"]):
                st.warning("Models missing. Run in terminal:")
                if not health["has_llm"]:
                    st.code(f"ollama pull {DEFAULT_LLM_MODEL}", language="bash")
                if not health["has_embedding"]:
                    st.code(f"ollama pull {DEFAULT_EMBEDDING_MODEL}", language="bash")
        else:
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 0.35rem;">
                    <span style="font-size: 0.8rem; font-weight: 600; color: #E2E8F0;">Local Engine</span>
                    <span style="font-size: 0.72rem; color: #EF4444; font-weight: 600;">● Offline</span>
                </div>
                <div style="font-size: 0.74rem; color: #94A3B8;">Start Ollama daemon: <code>ollama serve</code></div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # 4.3 PDF Document Ingestion
    st.markdown("<span style='font-size: 0.82rem; font-weight: 600; color: #E2E8F0;'>Document Ingestion</span>", unsafe_allow_html=True)
