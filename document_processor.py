"""Document processing module for Chat with Your Documents.

Beginner-friendly module explaining:
1. Document Loading: Reading PDF files and extracting text page-by-page with PyPDF.
2. Metadata Preservation: Tagging each page with its source filename and page number.
3. Edge Case Validation: Detecting scanned (image-only) or empty PDFs early.
4. Chunking: Splitting long pages into smaller pieces using RecursiveCharacterTextSplitter.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Tuple, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfStreamError

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MAX_UPLOAD_SIZE_MB,
    MIN_EXTRACTABLE_TEXT_LENGTH,
)


# -----------------------------------------------------------------------------
# Custom Error Types for Clear, Beginner-Friendly Feedback
# -----------------------------------------------------------------------------
class DocumentProcessingError(Exception):
    """Base exception for all document processing errors."""
    pass


class EmptyPDFError(DocumentProcessingError):
    """Raised when an uploaded PDF has no pages or zero text."""
    pass


class ScannedPDFError(DocumentProcessingError):
    """Raised when a PDF contains images/scans without a selectable text layer."""
    pass


class EncryptedPDFError(DocumentProcessingError):
    """Raised when an uploaded PDF is password protected."""
    pass


class FileSizeExceededError(DocumentProcessingError):
    """Raised when an uploaded PDF is larger than MAX_UPLOAD_SIZE_MB."""
    pass


@dataclass
class DocumentMetadataSummary:
    """A simple data container holding the results of document processing."""
    filename: str
    file_hash: str
    total_pages: int
    total_chunks: int
    total_characters: int
    chunks: List[Document]


# -----------------------------------------------------------------------------
# Helper Functions: Hashing & Size Checks
# -----------------------------------------------------------------------------
def compute_file_hash(file_bytes: bytes) -> str:
    """Generate a unique 16-character fingerprint from the file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def validate_file_size(file_bytes: bytes, max_mb: int = MAX_UPLOAD_SIZE_MB) -> None:
    """Ensure the file does not exceed the allowed upload limit."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise FileSizeExceededError(
            f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_mb} MB."
        )


def _read_bytes_from_source(file_source: Union[str, Path, BinaryIO, bytes]) -> Tuple[bytes, str]:
    """Helper to convert any file input (path, stream, or bytes) into raw bytes."""
    if isinstance(file_source, (str, Path)):
        source_path = Path(file_source)
        return source_path.read_bytes(), source_path.name
    elif isinstance(file_source, bytes):
        return file_source, "document.pdf"
    elif hasattr(file_source, "read"):
        content = file_source.read()
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        filename = getattr(file_source, "name", "document.pdf")
        return content, filename
    else:
        raise DocumentProcessingError(f"Unsupported input type: {type(file_source)}")


# -----------------------------------------------------------------------------
# Core Step 1: Extract Text & Preserve Page Metadata
# -----------------------------------------------------------------------------
def extract_pages_from_pdf(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: str = "document.pdf",
) -> Tuple[List[Document], str]:
    """
    Read a PDF file and convert each page into a LangChain Document.
    
    Why metadata matters in RAG:
    Every Document object stores both the text (page_content) and metadata
    (like source file name and page number). When the LLM answers later,
    we use this metadata to show exact page citations!
    """
    raw_bytes, detected_name = _read_bytes_from_source(file_source)
    if filename == "document.pdf" and detected_name != "document.pdf":
        filename = detected_name

    if not raw_bytes:
        raise EmptyPDFError("The provided file is empty (0 bytes).")

    validate_file_size(raw_bytes)
    file_hash = compute_file_hash(raw_bytes)

    # Open PDF using PyPDF
    try:
        pdf_reader = PdfReader(io.BytesIO(raw_bytes))
    except (EmptyFileError, PdfStreamError) as err:
        raise DocumentProcessingError(f"Cannot read PDF format: {err}") from err
    except Exception as err:
        raise DocumentProcessingError(f"Failed to open PDF: {err}") from err

    # Check for password protection
    if pdf_reader.is_encrypted:
        try:
            if pdf_reader.decrypt("") == 0:
                raise EncryptedPDFError(f"'{filename}' is password protected.")
        except Exception:
            raise EncryptedPDFError(f"'{filename}' is password protected.")

    total_pages = len(pdf_reader.pages)
    if total_pages == 0:
        raise EmptyPDFError(f"'{filename}' contains 0 pages.")

    page_documents: List[Document] = []
    total_characters = 0

    # Loop over every page (1-indexed for human readability)
    for page_number, page in enumerate(pdf_reader.pages, start=1):
        try:
            extracted_text = (page.extract_text() or "").strip()
        except Exception:
            extracted_text = ""

        total_characters += len(extracted_text)

        # Only create a document if the page contains text
        if extracted_text:
            page_documents.append(
                Document(
                    page_content=extracted_text,
                    metadata={
                        "source": filename,
                        "file_hash": file_hash,
                        "page": page_number,
                        "total_pages": total_pages,
                        "char_count": len(extracted_text),
                    },
                )
            )

    # Scanned PDF and Blank PDF Validation
    if total_characters == 0:
        raise EmptyPDFError(
            f"'{filename}' has {total_pages} page(s) but no extractable text could be found. "
            "It may be completely blank or contain only unreadable scanned images."
        )

    if total_characters < MIN_EXTRACTABLE_TEXT_LENGTH:
        raise ScannedPDFError(
            f"'{filename}' contains very little extractable text ({total_characters} characters). "
            "It appears to be a scanned document or image-only PDF. "
            "Chat with Your Documents requires selectable text."
        )

    return page_documents, file_hash


# -----------------------------------------------------------------------------
# Core Step 2: Split Documents into Overlapping Chunks
# -----------------------------------------------------------------------------
def split_documents_into_chunks(
    page_documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split long page documents into smaller, overlapping chunks.
    
    Why chunking is necessary:
    1. LLMs have context windows and can get confused by full books.
    2. Smaller chunks make retrieval much more accurate.
    3. Overlap ensures that a sentence split between chunks isn't cut in half!
    """
    if not page_documents:
        return []

    # Ensure overlap is always strictly smaller than chunk size
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    # LangChain text splitter that breaks on paragraphs, sentences, and words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(page_documents)

    # Attach helpful chunk identifiers for beginner inspection
    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "doc")
        page = chunk.metadata.get("page", 1)
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{source}_p{page}_c{index}"

    return chunks


# -----------------------------------------------------------------------------
# Core Step 3: End-to-End Orchestrator
# -----------------------------------------------------------------------------
def process_uploaded_pdf(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: str = "document.pdf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> DocumentMetadataSummary:
    """
    Orchestrate the full ingestion pipeline:
    1. Extract text and page numbers from PDF.
    2. Split pages into overlapping chunks.
    3. Return a clean summary ready for vector storage.
    """
    page_docs, file_hash = extract_pages_from_pdf(file_source, filename=filename)
    chunks = split_documents_into_chunks(
        page_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    total_pages = page_docs[0].metadata.get("total_pages", len(page_docs)) if page_docs else 0
    total_characters = sum(len(c.page_content) for c in chunks)

    return DocumentMetadataSummary(
        filename=filename,
        file_hash=file_hash,
        total_pages=total_pages,
        total_chunks=len(chunks),
        total_characters=total_characters,
        chunks=chunks,
    )
