"""Vector store management for Chat with Your Documents using ChromaDB and Ollama.

Beginner-friendly module explaining:
1. Embeddings: converting text into numerical vectors using Ollama (nomic-embed-text).
2. Vector Store: storing and searching vectors using ChromaDB.
3. Document Isolation: keeping each PDF in its own Chroma collection to prevent data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


def sanitize_collection_name(name_seed: str) -> str:
    """
    Convert a filename or hash into a safe ChromaDB collection name.
    
    Chroma naming rules:
    - Must be 3 to 63 characters long.
    - Can only contain letters, numbers, underscores, and hyphens.
    - Must start and end with an alphanumeric character.
    """
    # Replace any non-alphanumeric character (except hyphen/underscore) with underscore
    safe_chars = [c if (c.isalnum() or c in ("_", "-")) else "_" for c in name_seed]
    cleaned = "".join(safe_chars).strip("_-")

    # Prefix with 'doc_' to ensure it starts with letters
    collection_name = f"doc_{cleaned}"[:63]

    # Ensure valid length
    if len(collection_name) < 3:
        return "doc_default"
    return collection_name


def get_embeddings(
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> OllamaEmbeddings:
    """
    Create a local LangChain embeddings object powered by Ollama.
    This runs completely on your machine with zero cloud API keys.
    """
    return OllamaEmbeddings(
        model=model,
        base_url=base_url,
    )


class VectorStoreManager:
    """
    Manages local ChromaDB vector storage with document isolation.
    
    Why Document Isolation?
    When a user uploads 'contract.pdf', we index its chunks into a collection
    named after that file's unique hash. When they later upload 'notes.pdf',
    questions will only search 'notes.pdf' — preventing old chunks from leaking in!
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        # Local folder on disk where Chroma saves database files
        self.persist_dir = Path(persist_dir) if persist_dir else CHROMA_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.base_url = base_url
        self.embeddings = get_embeddings(model=self.embedding_model, base_url=self.base_url)

        # Chroma client for administrative tasks (like deleting or inspecting collections)
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

    def get_collection_name(self, file_hash: str) -> str:
        """Derive an isolated collection name from the document's content hash."""
        return sanitize_collection_name(f"doc_{file_hash}")

    def collection_exists(self, file_hash: str) -> bool:
        """Check if an isolated collection already exists and has indexed chunks."""
        collection_name = self.get_collection_name(file_hash)
        try:
            collection = self.chroma_client.get_collection(collection_name)
            return collection.count() > 0
        except Exception:
            return False

    def index_document(
        self,
        chunks: List[Document],
        file_hash: str,
    ) -> Chroma:
        """
        Step 1 in Vector Storage:
        Takes text chunks, generates vector embeddings using Ollama,
        and saves them into an isolated ChromaDB collection.
        """
        if not chunks:
            raise ValueError("Cannot index an empty list of document chunks.")

        collection_name = self.get_collection_name(file_hash)

        # Clear any prior collection with this name to avoid duplicate chunks
        self.delete_collection(file_hash)

        # Create the collection and index all chunks with embeddings
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=str(self.persist_dir),
            client=self.chroma_client,
        )

        return vector_store

    def get_vector_store(self, file_hash: str) -> Chroma:
        """Connect to an existing isolated Chroma vector store."""
        collection_name = self.get_collection_name(file_hash)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
            client=self.chroma_client,
        )

    def search_relevant_chunks(
        self,
        file_hash: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[Document, float]]:
        """
        Step 2 in Vector Storage (Retrieval):
        Converts the user question into an embedding vector, then compares it
        against all chunk vectors in this document to find the most relevant ones.
        Returns a list of (Document, distance_score) tuples.
        """
        vector_store = self.get_vector_store(file_hash)
        return vector_store.similarity_search_with_score(query=query, k=top_k)

    def delete_collection(self, file_hash: str) -> bool:
        """
        Safely deletes a document's collection.
        Called when a user clicks 'Remove File' or replaces a document.
        """
        collection_name = self.get_collection_name(file_hash)
        try:
            self.chroma_client.delete_collection(collection_name)
            return True
        except Exception:
            return False

    def clear_all(self) -> None:
        """Reset all local Chroma collections."""
        try:
            for col in self.chroma_client.list_collections():
                self.chroma_client.delete_collection(col.name)
        except Exception as e:
            logger.warning(f"Error resetting collections: {e}")
