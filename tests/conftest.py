"""Pytest fixtures and PDF synthesis helpers for Chat with Your Documents."""

import io
from typing import List
import pytest
from pypdf import PdfWriter


def make_pdf_bytes(page_texts: List[str]) -> bytes:
    """
    Synthesize valid PDF bytes with specific text on each page using pure Python.
    """
    objects = []
    # 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages root (placeholder)
    objects.append(b"")
    # 3: Font definition
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_ids = []
    curr_id = 4
    for text in page_texts:
        page_id = curr_id
        curr_id += 1
        content_id = curr_id
        curr_id += 1
        page_obj_ids.append(page_id)

        stream_data = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode("latin1")
        content_obj = (
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin1")
            + stream_data
            + b"\nendstream"
        )
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("latin1")
        objects.append(page_obj)
        objects.append(content_obj)

    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_texts)} >>".encode("latin1")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    xref = [0]
    for i, obj in enumerate(objects, 1):
        xref.append(out.tell())
        out.write(f"{i} 0 obj\n".encode("latin1") + obj + b"\nendobj\n")
    startxref = out.tell()
    out.write(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode("latin1"))
    for offset in xref[1:]:
        out.write(f"{offset:010d} 00000 n \n".encode("latin1"))
    out.write(
        f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF".encode(
            "latin1"
        )
    )
    return out.getvalue()


@pytest.fixture
def sample_valid_pdf_bytes() -> bytes:
    """A valid 2-page PDF with realistic text content."""
    return make_pdf_bytes(
        [
            "Chat with Your Documents is a privacy-first RAG application running on local Ollama.",
            "Chapter 2 explains vector embeddings using nomic-embed-text and ChromaDB storage.",
        ]
    )


@pytest.fixture
def sample_empty_pdf_bytes() -> bytes:
    """A valid PDF structure containing only a blank page with zero text."""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def sample_scanned_pdf_bytes() -> bytes:
    """A PDF with fewer than 15 characters, simulating a scanned or image page."""
    return make_pdf_bytes(["A"])
