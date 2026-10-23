"""Unit tests for document processing and PDF extraction."""

import io
import pytest

from document_processor import (
    EmptyPDFError,
    FileSizeExceededError,
    ScannedPDFError,
    extract_pages_from_pdf,
    process_uploaded_pdf,
    split_documents_into_chunks,
)
from tests.conftest import make_pdf_bytes


def test_extract_pages_valid_pdf(sample_valid_pdf_bytes):
    """Test extracting text and metadata from a valid multi-page PDF."""
    docs, file_hash = extract_pages_from_pdf(sample_valid_pdf_bytes, filename="guide.pdf")

    assert len(docs) == 2
    assert len(file_hash) == 16

    # Verify Page 1 metadata
    assert docs[0].metadata["source"] == "guide.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 2
    assert "privacy-first RAG" in docs[0].page_content

    # Verify Page 2 metadata
    assert docs[1].metadata["source"] == "guide.pdf"
    assert docs[1].metadata["page"] == 2
    assert docs[1].metadata["total_pages"] == 2
    assert "Chapter 2" in docs[1].page_content


def test_extract_pages_empty_bytes():
    """Test that 0-byte input raises EmptyPDFError."""
    with pytest.raises(EmptyPDFError):
        extract_pages_from_pdf(b"", filename="empty.pdf")


def test_extract_pages_blank_pdf(sample_empty_pdf_bytes):
    """Test that a blank PDF page with zero extractable text raises EmptyPDFError."""
    with pytest.raises(EmptyPDFError) as exc_info:
        extract_pages_from_pdf(sample_empty_pdf_bytes, filename="blank.pdf")
    assert "no extractable text" in str(exc_info.value)


def test_extract_pages_scanned_pdf(sample_scanned_pdf_bytes):
    """Test that an image-only / scanned PDF with negligible text raises ScannedPDFError."""
    with pytest.raises(ScannedPDFError) as exc_info:
        extract_pages_from_pdf(sample_scanned_pdf_bytes, filename="scanned.pdf")
    assert "scanned document or image-only" in str(exc_info.value)


def test_file_size_exceeded():
    """Test that oversized files raise FileSizeExceededError."""
    huge_bytes = b"%PDF-1.4 " + b"0" * (26 * 1024 * 1024)
    with pytest.raises(FileSizeExceededError):
        extract_pages_from_pdf(huge_bytes, filename="huge.pdf")


def test_split_documents_into_chunks(sample_valid_pdf_bytes):
    """Test splitting page documents into smaller overlapping chunks."""
    page_docs, _ = extract_pages_from_pdf(sample_valid_pdf_bytes, filename="doc.pdf")
    chunks = split_documents_into_chunks(page_docs, chunk_size=50, chunk_overlap=10)

    assert len(chunks) >= len(page_docs)
    for idx, chunk in enumerate(chunks):
        assert "chunk_index" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["source"] == "doc.pdf"


def test_process_uploaded_pdf_summary(sample_valid_pdf_bytes):
    """Test the complete upload pipeline metadata summary."""
    summary = process_uploaded_pdf(
        file_source=io.BytesIO(sample_valid_pdf_bytes),
        filename="rag_paper.pdf",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert summary.filename == "rag_paper.pdf"
    assert summary.total_pages == 2
    assert summary.total_chunks > 0
    assert summary.total_characters > 0
    assert len(summary.chunks) == summary.total_chunks
