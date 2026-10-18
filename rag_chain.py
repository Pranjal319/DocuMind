"""RAG pipeline module for Chat with Your Documents.

Beginner-friendly module explaining:
1. Retrieval: Querying ChromaDB for the top-k most relevant chunks.
2. Prompt Construction: Combining retrieved context and user question into a grounded prompt.
3. Local Generation: Passing the prompt to local ChatOllama (llama3.2:3b).
4. Citations: Extracting page numbers and source passages for transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    OLLAMA_BASE_URL,
    RAG_SYSTEM_PROMPT,
)
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Containers: Citations and Responses
# -----------------------------------------------------------------------------
@dataclass
class Citation:
    """Represents a single retrieved passage cited in the answer."""
    source: str
    page: int
    total_pages: int
    chunk_id: str
    content: str
    relevance_score: Optional[float] = None


@dataclass
class RAGResponse:
    """The final structured answer from the RAG pipeline."""
    answer: str
    citations: List[Citation]
    missing_context: bool = False
    model_used: str = DEFAULT_LLM_MODEL


# -----------------------------------------------------------------------------
# System Check: Verify Local Ollama
# -----------------------------------------------------------------------------
def check_ollama_health(
    base_url: str = OLLAMA_BASE_URL,
    llm_model: str = DEFAULT_LLM_MODEL,
    embed_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Dict[str, Any]:
    """
    Check if Ollama is running on localhost and verify required models are downloaded.
    Never calls external cloud APIs.
    """
    status: Dict[str, Any] = {
        "connected": False,
        "base_url": base_url,
        "available_models": [],
        "has_llm": False,
        "has_embedding": False,
        "error_message": None,
    }

    try:
        # Query Ollama's local tags API
        response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        if response.status_code == 200:
            status["connected"] = True
            models_info = response.json().get("models", [])
            installed_names = [m.get("name", "") for m in models_info]
            status["available_models"] = installed_names

            # Simple helper: check if model name (e.g. 'llama3.2') is installed
            def is_installed(target_model: str) -> bool:
                base = target_model.split(":")[0].lower()
                return any(base in name.lower() for name in installed_names)

            status["has_llm"] = is_installed(llm_model)
            status["has_embedding"] = is_installed(embed_model)
        else:
            status["error_message"] = f"Ollama returned HTTP status {response.status_code}."
    except requests.exceptions.ConnectionError:
        status["error_message"] = f"Cannot reach Ollama at {base_url}. Is Ollama running?"
    except Exception as err:
        status["error_message"] = f"Ollama connection error: {err}"

    return status


# -----------------------------------------------------------------------------
# Helper: Format Context & Build Citations
# -----------------------------------------------------------------------------
def format_context_with_citations(
    retrieved_chunks: List[Tuple[Document, float]],
) -> Tuple[str, List[Citation]]:
    """
    Format retrieved chunks into readable text for the LLM prompt,
    and create structured Citation objects for the UI.
    """
    if not retrieved_chunks:
        return "", []

    text_sections: List[str] = []
    citations: List[Citation] = []

    for index, (chunk_doc, score) in enumerate(retrieved_chunks, start=1):
        source_file = chunk_doc.metadata.get("source", "document.pdf")
        page_num = chunk_doc.metadata.get("page", 1)
        total_pages = chunk_doc.metadata.get("total_pages", 1)
        chunk_id = chunk_doc.metadata.get("chunk_id", f"{source_file}_p{page_num}_c{index}")
        passage_text = chunk_doc.page_content.strip()

        # Format header so the LLM knows which page this text came from
        header = f"[Excerpt {index} | Source: {source_file} | Page: {page_num} of {total_pages}]"
        text_sections.append(f"{header}\n{passage_text}")

        # Store citation details for UI rendering
        citations.append(
            Citation(
                source=source_file,
                page=page_num,
                total_pages=total_pages,
                chunk_id=chunk_id,
                content=passage_text,
                relevance_score=round(float(score), 4) if score is not None else None,
            )
        )

    full_context_text = "\n\n".join(text_sections)
    return full_context_text, citations


# -----------------------------------------------------------------------------
# The RAG Pipeline: Retrieval + Prompt + Generation
# -----------------------------------------------------------------------------
class RAGPipeline:
    """
    Coordinates the 4 main stages of RAG:
    1. Retrieve relevant passages using vector similarity search.
    2. Format passages into context with source page labels.
    3. Construct a strictly grounded prompt.
    4. Generate the final answer using local ChatOllama.
    """

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.vector_store_manager = vector_store_manager
        self.llm_model = llm_model
        self.temperature = temperature
        self.base_url = base_url

        # LangChain PromptTemplate that injects {context} and {question}
        self.prompt_template = PromptTemplate(
            template=RAG_SYSTEM_PROMPT,
            input_variables=["context", "question"],
        )

        # Local Ollama chat model
        self.llm = ChatOllama(
            model=self.llm_model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def answer_question(
        self,
        file_hash: str,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> RAGResponse:
        """
        Execute the complete RAG cycle for a user's question:
        
        Step 1: Validate question.
        Step 2: Retrieve top_k relevant chunks from the isolated document collection.
        Step 3: Format the context text and collect citation objects.
        Step 4: Build the prompt using the prompt template.
        Step 5: Call ChatOllama to generate the grounded response.
        """
        clean_question = question.strip()
        if not clean_question:
            return RAGResponse(
                answer="Please enter a valid, non-empty question.",
                citations=[],
                missing_context=True,
                model_used=self.llm_model,
            )

        # Step 2: Retrieve relevant chunks
        retrieved_items = self.vector_store_manager.search_relevant_chunks(
            file_hash=file_hash,
            query=clean_question,
            top_k=top_k,
        )

        if not retrieved_items:
            return RAGResponse(
                answer="I could not find any relevant information in the uploaded document context for your question.",
                citations=[],
                missing_context=True,
                model_used=self.llm_model,
            )

        # Step 3: Format context & assemble citations
        context_text, citations = format_context_with_citations(retrieved_items)

        # Step 4: Construct the grounded prompt
        prompt_string = self.prompt_template.format(
            context=context_text,
            question=clean_question,
        )

        # Step 5: Generate answer via local LLM
        try:
            llm_result = self.llm.invoke(prompt_string)
            answer_text = llm_result.content.strip()
        except Exception as err:
            logger.error(f"Ollama generation failed: {err}")
            raise RuntimeError(
                f"Ollama inference error: {err}. Ensure Ollama is running and has {self.llm_model}."
            ) from err

        # Check if the LLM concluded the answer wasn't present in context
        missing_phrases = [
            "could not find",
            "not found in the provided",
            "not mentioned in",
            "does not contain",
            "no mention of",
            "insufficient information",
        ]
        is_missing = any(phrase in answer_text.lower() for phrase in missing_phrases)

        return RAGResponse(
            answer=answer_text,
            citations=citations,
            missing_context=is_missing,
            model_used=self.llm_model,
        )
