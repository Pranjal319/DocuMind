"""Unit and integration tests for RAG pipeline, grounding, and citation extraction."""

import shutil
import tempfile
from pathlib import Path
import pytest
from langchain_core.documents import Document

from rag_chain import (
    RAGPipeline,
    check_ollama_health,
    format_context_with_citations,
)
from vector_store import VectorStoreManager


def test_ollama_health_check():
    """Verify that local Ollama health check responds and detects models."""
    health = check_ollama_health()
    assert isinstance(health, dict)
    assert "connected" in health
    assert health["connected"] is True
    assert health["has_llm"] is True
    assert health["has_embedding"] is True


def test_format_context_with_citations():
    """Verify that retrieved document chunks are correctly formatted with metadata citations."""
    doc1 = Document(
        page_content="LangChain provides modular abstractions for LLM applications.",
        metadata={"source": "tech_spec.pdf", "page": 3, "total_pages": 10, "chunk_id": "c1"},
    )
    doc2 = Document(
        page_content="ChromaDB offers lightweight embedding storage.",
        metadata={"source": "tech_spec.pdf", "page": 5, "total_pages": 10, "chunk_id": "c2"},
    )

    retrieved = [(doc1, 0.12), (doc2, 0.45)]
    context_str, citations = format_context_with_citations(retrieved)

    # Check context formatting
    assert "[Excerpt 1 | Source: tech_spec.pdf | Page: 3 of 10]" in context_str
    assert "[Excerpt 2 | Source: tech_spec.pdf | Page: 5 of 10]" in context_str
    assert "LangChain provides" in context_str

    # Check citations
    assert len(citations) == 2
    assert citations[0].source == "tech_spec.pdf"
    assert citations[0].page == 3
    assert citations[0].total_pages == 10
    assert citations[0].relevance_score == 0.12
    assert citations[1].page == 5


def test_empty_query_handling():
    """Verify that blank queries return an immediate prompt response without LLM calls."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        manager = VectorStoreManager(persist_dir=tmp_dir)
        pipeline = RAGPipeline(vector_store_manager=manager)
        response = pipeline.answer_question(file_hash="dummy", question="   ")
        assert response.missing_context is True
        assert "valid, non-empty" in response.answer.lower()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_rag_pipeline_grounded_answer_and_citations():
    """
    Integration test: Query an indexed document using local llama3.2:3b and nomic-embed-text.
    Verifies that the generated answer is grounded and contains citations.
    """
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        manager = VectorStoreManager(persist_dir=tmp_dir)
        pipeline = RAGPipeline(vector_store_manager=manager)

        doc = Document(
            page_content="The Project Falcon spacecraft launched successfully on May 14, 2024, powered by liquid methane engines.",
            metadata={"source": "mission_report.pdf", "page": 7, "total_pages": 12, "chunk_id": "falcon_c0"},
        )
        file_hash = "falcon123"
        manager.index_document([doc], file_hash=file_hash)

        # Ask a question directly answered by the document
        response = pipeline.answer_question(
            file_hash=file_hash,
            question="When did the Project Falcon spacecraft launch and what engines did it use?",
            top_k=1,
        )

        assert len(response.citations) == 1
        assert response.citations[0].page == 7
        assert response.citations[0].source == "mission_report.pdf"
        assert len(response.answer) > 0
        # The answer should mention May 14 or 2024 or methane
        assert any(
            kw in response.answer.lower()
            for kw in ["may", "2024", "methane", "falcon"]
        )

        # Ask a question NOT answered by the document
        unrelated_resp = pipeline.answer_question(
            file_hash=file_hash,
            question="What is the capital of Australia and who was its first prime minister?",
            top_k=1,
        )

        # Should indicate not found or missing
        assert (
            unrelated_resp.missing_context
            or "could not find" in unrelated_resp.answer.lower()
            or "not mentioned" in unrelated_resp.answer.lower()
            or "not" in unrelated_resp.answer.lower()
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
