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
    uploaded_file = st.file_uploader(
        "Upload text-based PDF",
        type=["pdf"],
        help=f"Limit: {MAX_UPLOAD_SIZE_MB}MB. Scanned image PDFs are rejected.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = compute_file_hash(file_bytes)
        current_hash = (
            st.session_state.active_doc.get("file_hash")
            if st.session_state.active_doc
            else None
        )

        if current_hash != file_hash:
            try:
                with st.spinner("Extracting text and chunking PDF..."):
                    summary = process_uploaded_pdf(
                        file_source=file_bytes,
                        filename=uploaded_file.name,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    )

                with st.spinner("Indexing vector embeddings in ChromaDB..."):
                    st.session_state.vector_manager.index_document(
                        chunks=summary.chunks,
                        file_hash=summary.file_hash,
                    )

                st.session_state.active_doc = {
                    "filename": summary.filename,
                    "file_hash": summary.file_hash,
                    "total_pages": summary.total_pages,
                    "total_chunks": summary.total_chunks,
                    "total_characters": summary.total_characters,
                    "sample_chunks": summary.chunks[:3],
                }
                st.session_state.messages = []
                st.success(f"Indexed: {summary.filename}")

            except (ScannedPDFError, EmptyPDFError, EncryptedPDFError, FileSizeExceededError, DocumentProcessingError) as err:
                st.error(f"Ingestion Error: {err}")
            except Exception as err:
                logger.error(f"Error processing PDF: {err}", exc_info=True)
                st.error(f"Unexpected error: {err}")

    # 4.4 Active Document Summary & Controls
    if st.session_state.active_doc:
        doc = st.session_state.active_doc
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="font-size: 0.78rem; font-weight: 600; color: #A5B4FC; margin-bottom: 0.35rem; word-break: break-all;">
                    📄 {doc['filename']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Pages", doc["total_pages"])
            c2.metric("Chunks", doc["total_chunks"])
            c3.metric("Size", f"{doc['total_characters']//1000}k")

            col_replace, col_reset = st.columns(2)
            with col_replace:
                if st.button("🔄 Replace", use_container_width=True, help="Remove document and upload new PDF"):
                    st.session_state.vector_manager.delete_collection(doc["file_hash"])
                    st.session_state.active_doc = None
                    st.session_state.messages = []
                    st.rerun()
            with col_reset:
                if st.button("🧹 Clear Chat", use_container_width=True, help="Reset conversation history"):
                    st.session_state.messages = []
                    st.rerun()

        # Chunk Inspector
        with st.expander("🔍 Chunk Inspector"):
            st.caption("Inspect partitioned document chunks:")
            for idx, chunk in enumerate(doc["sample_chunks"], start=1):
                st.markdown(f"**Chunk {idx}** • Page {chunk.metadata.get('page')}")
                preview = chunk.page_content[:170] + "..." if len(chunk.page_content) > 170 else chunk.page_content
                st.code(preview, language="text")

    # 4.5 Retrieval & Model Settings
    with st.expander("⚙️ Retrieval Parameters"):
        top_k = st.slider("Top Chunks (k)", min_value=1, max_value=8, value=DEFAULT_TOP_K)
        temp = st.slider("Temperature", min_value=0.0, max_value=0.7, value=DEFAULT_TEMPERATURE, step=0.05)
        st.session_state.rag_pipeline.temperature = temp


# -----------------------------------------------------------------------------
# 5. Main Area: Brand Header & Welcome State
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-title">
        <span class="brand-spark">✦</span> DocuMind
    </div>
    <div class="brand-tagline">
        Your documents. Your knowledge. One conversation.
    </div>
    <div class="badge-row">
        <span class="saas-badge">🔒 100% Private & Local</span>
        <span class="saas-badge">🦙 Llama 3.2 (3B)</span>
        <span class="saas-badge">⚡ Nomic Embed</span>
        <span class="saas-badge">📦 ChromaDB Storage</span>
        <span class="saas-badge">🛡️ Document Isolated</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Active document status bar
if st.session_state.active_doc:
    doc = st.session_state.active_doc
    st.markdown(
        f"""
        <div class="doc-banner">
            <div>
                <strong>Active Context:</strong> <code>{doc['filename']}</code>
                &nbsp;|&nbsp; {doc['total_pages']} pages
                &nbsp;|&nbsp; {doc['total_chunks']} chunks
            </div>
            <div class="doc-banner-status">
                <span>●</span> Indexed & Ready
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Minimal, SaaS-style Empty/Welcome State
    with st.container(border=True):
        st.markdown(
            """
            <div style="padding: 0.5rem 0.2rem;">
                <h3 style="margin-top: 0; font-size: 1.15rem; font-weight: 600; color: #F8FAFC;">
                    Welcome to DocuMind
                </h3>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.85rem;">
                    DocuMind enables grounded, contextual conversations with text-based PDF documents
                    using local AI inference. Everything executes on your device with zero cloud dependencies.
                </p>
                <div style="background: rgba(99, 102, 241, 0.08); border: 1px dashed rgba(99, 102, 241, 0.35); border-radius: 6px; padding: 0.65rem 0.85rem; margin-bottom: 1rem; color: #E2E8F0; font-size: 0.84rem;">
                    👈 <strong>Get Started:</strong> Upload a text-based PDF in the sidebar to begin analyzing documents.
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 0.75rem; border-radius: 6px;">
                        <span style="color: #818CF8; font-weight: 600;">1. Ingest PDF</span><br>
                        <span style="font-size: 0.78rem; color: #94A3B8;">Upload a PDF in the sidebar to extract text and preserve page numbers.</span>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 0.75rem; border-radius: 6px;">
                        <span style="color: #818CF8; font-weight: 600;">2. Embed & Index</span><br>
                        <span style="font-size: 0.78rem; color: #94A3B8;">Chunks are embedded with nomic-embed-text into an isolated collection.</span>
                    </div>
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 0.75rem; border-radius: 6px;">
                        <span style="color: #818CF8; font-weight: 600;">3. Grounded Q&A</span><br>
                        <span style="font-size: 0.78rem; color: #94A3B8;">Ask questions to retrieve relevant passages and generate verified answers.</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 6. Interactive Suggested Questions (When document is active & chat is fresh)
# -----------------------------------------------------------------------------
selected_example_question = None

if st.session_state.active_doc and not st.session_state.messages:
    st.markdown("<span style='font-size: 0.8rem; font-weight: 600; color: #94A3B8;'>Suggested Questions:</span>", unsafe_allow_html=True)
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("📌 Summarize main topic", use_container_width=True):
            selected_example_question = "Can you summarize the main topic and purpose of this document?"
    with col_ex2:
        if st.button("🔍 Key findings & takeaways", use_container_width=True):
            selected_example_question = "What are the most important conclusions, findings, or takeaways in this document?"
    with col_ex3:
        if st.button("❓ Dates, entities & metrics", use_container_width=True):
            selected_example_question = "What important dates, people, organizations, or metrics are discussed in this document?"


# -----------------------------------------------------------------------------
# 7. Render Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []), expanded=False)


# -----------------------------------------------------------------------------
# 8. User Input & Grounded Generation
# -----------------------------------------------------------------------------
user_typed_question = st.chat_input(
    f"Ask a question about {st.session_state.active_doc['filename']}..."
    if st.session_state.active_doc
    else "Upload a PDF in the sidebar to begin conversation..."
)

user_question = user_typed_question or selected_example_question

if user_question:
    if not st.session_state.active_doc:
        st.warning("Please upload a PDF document in the sidebar first.")
    elif not health["connected"]:
        st.error("Ollama service is not running. Please launch Ollama (`ollama serve`) and retry.")
    else:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate assistant grounded response
        with st.chat_message("assistant"):
            active_hash = st.session_state.active_doc["file_hash"]
            try:
                with st.spinner("Searching document vectors & generating grounded answer..."):
                    rag_result = st.session_state.rag_pipeline.answer_question(
                        file_hash=active_hash,
                        question=user_question,
                        top_k=top_k,
                    )

                st.markdown(rag_result.answer)

                if rag_result.missing_context:
                    st.markdown(
                        """
                        <div class="grounding-alert">
                            ℹ️ <b>Grounding Notice:</b> The local model indicated that the answer is not
                            fully stated in the retrieved excerpts.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Convert Citation objects to serializable dictionaries for session history
                serialized_citations = [
                    {
                        "source": c.source,
                        "page": c.page,
                        "total_pages": c.total_pages,
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "score": c.relevance_score,
                    }
                    for c in rag_result.citations
                ]
                render_citations(serialized_citations, expanded=True)

                # Save assistant response to conversation history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": rag_result.answer,
                        "citations": serialized_citations,
                        "missing_context": rag_result.missing_context,
                    }
                )

            except Exception as err:
                logger.error(f"Answer generation error: {err}", exc_info=True)
                error_text = f"❌ Error generating response: {err}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text, "citations": []})
