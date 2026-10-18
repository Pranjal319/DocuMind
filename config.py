"""Application configuration and constants for Chat with Your Documents."""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Ollama Local Configuration
# Strictly free and local - connects to local Ollama daemon
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

# Document Processing Defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
MAX_UPLOAD_SIZE_MB = 25
MIN_EXTRACTABLE_TEXT_LENGTH = 30  # Chars across document below which PDF is flagged as scanned/empty

# Retrieval & Generation Defaults
DEFAULT_TOP_K = 4
DEFAULT_TEMPERATURE = 0.1  # Low temperature for deterministic, factual grounding

# Strict Grounded RAG Prompt Template
RAG_SYSTEM_PROMPT = """You are a precise, reliable document assistant called "DocuMind".
Your task is to answer the user's question based EXCLUSIVELY on the provided document excerpts.

STRICT GROUNDING RULES:
1. Rely ONLY on the information present in the excerpts below. Do NOT use outside knowledge, prior knowledge, or speculation.
2. If the excerpts do not contain enough facts to answer the question accurately, you MUST reply:
   "I could not find the answer in the provided document context."
3. Do NOT make up facts or extrapolate beyond what is explicitly stated in the context.
4. When relevant, reference the source page number(s) mentioned in the context (e.g., "[Page X]").
5. Keep answers clear, well-structured, and concise.

Context excerpts:
-----------------
{context}
-----------------

Question: {question}

Helpful & Grounded Answer:"""
