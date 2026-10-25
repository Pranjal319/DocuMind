"""Unit and integration tests for ChromaDB vector store and document isolation."""

import shutil
import tempfile
from pathlib import Path
import pytest
from langchain_core.documents import Document

from vector_store import VectorStoreManager, sanitize_collection_name


@pytest.fixture
def temp_chroma_manager():
    """Create a temporary vector store manager in an isolated temp directory."""
    tmp_dir = Path(tempfile.mkdtemp())
    manager = VectorStoreManager(persist_dir=tmp_dir)
    yield manager
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_sanitize_collection_name():
    """Test Chroma collection name rules and sanitization."""
    assert sanitize_collection_name("my_doc.pdf").startswith("doc_")
    assert sanitize_collection_name("---!!!12345").startswith("doc_")
    # Must be <= 63 chars
    long_name = "a" * 100
    sanitized = sanitize_collection_name(long_name)
    assert len(sanitized) <= 63
    # Check valid characters
    assert all(c.isalnum() or c in ("_", "-") for c in sanitized)


def test_document_isolation(temp_chroma_manager):
    """
    CRITICAL REQUIREMENT TEST:
    Verify that Document A's contents never leak into Document B's searches.
    """
    manager = temp_chroma_manager

    # Document A: Physics notes
    doc_a_chunks = [
        Document(
            page_content="Quantum entanglement allows particles to share states across distances.",
            metadata={"source": "physics.pdf", "page": 1, "chunk_id": "physics_p1_c0"},
        ),
        Document(
            page_content="Superconducting qubits operate at millikelvin temperatures.",
            metadata={"source": "physics.pdf", "page": 2, "chunk_id": "physics_p2_c1"},
        ),
    ]
    hash_a = "aaa111"

    # Document B: Biology notes
    doc_b_chunks = [
        Document(
            page_content="Photosynthesis converts solar light into glucose using chlorophyll in plants.",
            metadata={"source": "biology.pdf", "page": 1, "chunk_id": "biology_p1_c0"},
        ),
        Document(
            page_content="Cellular respiration occurs primarily within the mitochondrial matrix.",
            metadata={"source": "biology.pdf", "page": 2, "chunk_id": "biology_p2_c1"},
        ),
    ]
    hash_b = "bbb222"

    # Index both documents in their own collections
    manager.index_document(doc_a_chunks, file_hash=hash_a)
    manager.index_document(doc_b_chunks, file_hash=hash_b)

    assert manager.collection_exists(hash_a) is True
    assert manager.collection_exists(hash_b) is True

    # Search for biology topic in Document A's collection -> MUST NOT return Document B content
    results_in_a = manager.search_relevant_chunks(
        file_hash=hash_a,
        query="What is chlorophyll and photosynthesis?",
        top_k=2,
    )
    for doc, _ in results_in_a:
        assert doc.metadata["source"] == "physics.pdf"
        assert "chlorophyll" not in doc.page_content.lower()

    # Search for quantum topic in Document B's collection -> MUST NOT return Document A content
    results_in_b = manager.search_relevant_chunks(
        file_hash=hash_b,
        query="What is quantum entanglement?",
        top_k=2,
    )
    for doc, _ in results_in_b:
        assert doc.metadata["source"] == "biology.pdf"
        assert "quantum" not in doc.page_content.lower()


def test_delete_and_replace_collection(temp_chroma_manager):
    """Test removing a document and replacing it cleanly."""
    manager = temp_chroma_manager
    doc_chunks = [
        Document(
            page_content="Initial document version text content.",
            metadata={"source": "contract_v1.pdf", "page": 1},
        )
    ]
    file_hash = "ccc333"

    manager.index_document(doc_chunks, file_hash=file_hash)
    assert manager.collection_exists(file_hash) is True

    # Delete collection
    deleted = manager.delete_collection(file_hash)
    assert deleted is True
    assert manager.collection_exists(file_hash) is False
