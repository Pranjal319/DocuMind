"""Audit tests for Chat with Your Documents using a sample document with known facts.

Covers:
1. Question whose answer exists in the document.
2. Question whose answer does NOT exist in the document (document-adjacent).
3. Question about a completely different topic.

Includes:
- Fast mocked tests (can run without running Ollama / downloading models).
- Real end-to-end tests against local Ollama (when service is available).
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_chain import RAGPipeline, check_ollama_health
from vector_store import VectorStoreManager

# -----------------------------------------------------------------------------
# Known Sample Facts for Testing
# -----------------------------------------------------------------------------
SAMPLE_FACTS_PAGE_1 = (
    "Apollo 11 launched from Kennedy Space Center on July 16, 1969. "
    "The crew consisted of Commander Neil Armstrong, Command Module Pilot Michael Collins, "
    "and Lunar Module Pilot Buzz Aldrin."
)

SAMPLE_FACTS_PAGE_2 = (
    "On July 20, 1969, the Lunar Module Eagle touched down at the Sea of Tranquility. "
    "Neil Armstrong stepped onto the lunar surface and spoke the famous words: "
    "'That's one small step for man, one giant leap for mankind.'"
)


@pytest.fixture
def sample_known_document_chunks():
    """Two chunks from a known sample document across two distinct pages."""
    return [
        Document(
            page_content=SAMPLE_FACTS_PAGE_1,
            metadata={
                "source": "apollo_11_mission.pdf",
                "page": 1,
                "total_pages": 2,
                "chunk_id": "apollo_11_mission.pdf_p1_c0",
            },
        ),
        Document(
            page_content=SAMPLE_FACTS_PAGE_2,
            metadata={
                "source": "apollo_11_mission.pdf",
                "page": 2,
                "total_pages": 2,
                "chunk_id": "apollo_11_mission.pdf_p2_c1",
            },
        ),
    ]


# -----------------------------------------------------------------------------
# 1. Mocked Unit Tests (Runs without local Ollama daemon)
# -----------------------------------------------------------------------------
def test_audit_mocked_answer_exists(sample_known_document_chunks):
    """
    Test 1: Question whose answer exists in the document (Mocked).
    Verifies retrieval, prompt assembly, and citation metadata matching.
    """
    mock_vector_manager = MagicMock(spec=VectorStoreManager)
    # Simulate retriever returning the matching chunk from page 1
    matching_chunk = sample_known_document_chunks[0]
    mock_vector_manager.search_relevant_chunks.return_value = [(matching_chunk, 0.15)]

    pipeline = RAGPipeline(vector_store_manager=mock_vector_manager)

    # Mock ChatOllama's invoke to simulate grounded response
    mock_llm_response = AIMessage(
        content="The Commander of Apollo 11 was Neil Armstrong, and it launched on July 16, 1969 [Page 1]."
    )
    pipeline.llm = MagicMock()
    pipeline.llm.invoke.return_value = mock_llm_response

    response = pipeline.answer_question(
        file_hash="test_apollo_hash",
        question="Who was the Commander of Apollo 11 and when did it launch?",
        top_k=1,
    )

    # Verify retrieval was invoked with correct parameters
    mock_vector_manager.search_relevant_chunks.assert_called_once_with(
        file_hash="test_apollo_hash",
        query="Who was the Commander of Apollo 11 and when did it launch?",
        top_k=1,
    )

    # Verify LLM was invoked with prompt containing context and question
    call_prompt = pipeline.llm.invoke.call_args[0][0]
    assert "July 16, 1969" in call_prompt
    assert "Who was the Commander" in call_prompt

    # Verify output correctness
    assert response.missing_context is False
    assert "Neil Armstrong" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].source == "apollo_11_mission.pdf"
    assert response.citations[0].page == 1
    assert response.citations[0].total_pages == 2


def test_audit_mocked_answer_does_not_exist(sample_known_document_chunks):
    """
    Test 2: Question whose answer does NOT exist in the document (Mocked).
    Verifies that the missing_context flag is set when the LLM states not found.
    """
    mock_vector_manager = MagicMock(spec=VectorStoreManager)
    # The retriever returns chunks, but they do NOT contain the answer
    mock_vector_manager.search_relevant_chunks.return_value = [
        (sample_known_document_chunks[0], 0.65)
    ]

    pipeline = RAGPipeline(vector_store_manager=mock_vector_manager)

    # Simulate strictly grounded LLM following system prompt rule #2
    mock_llm_response = AIMessage(
        content="I could not find the answer in the provided document context."
    )
    pipeline.llm = MagicMock()
    pipeline.llm.invoke.return_value = mock_llm_response

    response = pipeline.answer_question(
        file_hash="test_apollo_hash",
        question="How many kilograms of lunar rocks did Apollo 11 bring back?",
        top_k=1,
    )

    assert response.missing_context is True
    assert "could not find" in response.answer.lower()


def test_audit_mocked_different_topic(sample_known_document_chunks):
    """
    Test 3: Question about a completely different topic (Mocked).
    Verifies handling of completely unrelated user queries.
    """
    mock_vector_manager = MagicMock(spec=VectorStoreManager)
    # When no chunks are relevant or returned
    mock_vector_manager.search_relevant_chunks.return_value = []

    pipeline = RAGPipeline(vector_store_manager=mock_vector_manager)

    response = pipeline.answer_question(
        file_hash="test_apollo_hash",
        question="What is the recipe for baking chocolate chip cookies?",
        top_k=2,
    )

    assert response.missing_context is True
    assert "could not find any relevant information" in response.answer.lower()
    assert len(response.citations) == 0


# -----------------------------------------------------------------------------
# 2. Real End-to-End Tests (Uses active local Ollama models)
# -----------------------------------------------------------------------------
def test_audit_real_end_to_end_known_document(sample_known_document_chunks):
    """
    Real End-to-End Test: Indexes the known sample document in ChromaDB using local
    nomic-embed-text and answers all three question types using local llama3.2:3b.
    """
    health = check_ollama_health()
    if not (health["connected"] and health["has_llm"] and health["has_embedding"]):
        pytest.skip("Local Ollama or required models not available for live test.")

    temp_db_dir = Path(tempfile.mkdtemp())
    try:
        manager = VectorStoreManager(persist_dir=temp_db_dir)
        pipeline = RAGPipeline(vector_store_manager=manager)

        file_hash = "apollo_known_hash_99"
        manager.index_document(sample_known_document_chunks, file_hash=file_hash)

        # 1. Question with answer present
        q1 = "Who was the Lunar Module Pilot on Apollo 11 and where did it land?"
        res1 = pipeline.answer_question(file_hash=file_hash, question=q1, top_k=2)

        assert len(res1.citations) > 0
        assert res1.citations[0].source == "apollo_11_mission.pdf"
        assert any("aldrin" in res1.answer.lower() or "tranquility" in res1.answer.lower() for _ in [1])

        # 2. Question with answer NOT present
        q2 = "What was the exact price of gold in US dollars during July 1969?"
        res2 = pipeline.answer_question(file_hash=file_hash, question=q2, top_k=2)

        # Should indicate not found / missing context
        assert (
            res2.missing_context
            or "could not find" in res2.answer.lower()
            or "not mentioned" in res2.answer.lower()
            or "not found" in res2.answer.lower()
            or "does not contain" in res2.answer.lower()
        )

        # 3. Question about an unrelated topic
        q3 = "How do you cultivate Japanese bonsai trees indoors?"
        res3 = pipeline.answer_question(file_hash=file_hash, question=q3, top_k=1)

        assert (
            res3.missing_context
            or "could not find" in res3.answer.lower()
            or "not" in res3.answer.lower()
        )

    finally:
        shutil.rmtree(temp_db_dir, ignore_errors=True)
