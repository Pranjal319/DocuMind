# ✦ DocuMind

> *"Your documents. Your knowledge. One conversation."*

**DocuMind** is a modern, developer-grade **Retrieval-Augmented Generation (RAG)** application. Upload text-based PDF documents and ask questions about their contents. The system retrieves relevant passages from your uploaded PDF and uses a local LLM to generate factual, grounded answers with direct source citations and page numbers.

---

## 🌟 Key Features

- **🔒 100% Free & Local**: Zero API keys, zero paid cloud services (no OpenAI, no Gemini, no paid embeddings). Everything runs on your machine.
- **🦙 Local LLM**: Powered by Ollama running [`llama3.2:3b`](https://ollama.com/library/llama3.2).
- **⚡ Local Vector Embeddings**: Uses Ollama running [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) (768-dimensional local embeddings).
- **📦 Persistent Vector Storage**: Stores chunks and embeddings locally using [ChromaDB](https://www.trychroma.com/).
- **🛡️ Strict Document Isolation**: Every uploaded document is indexed in its own dedicated collection keyed by content hash (`doc_<hash>`). Eliminates cross-document context leakage.
- **📑 Verifiable Page Citations**: Answers display exact source filenames, page numbers, and retrieved excerpts in collapsible UI cards.
- **🚫 Anti-Hallucination Grounding**: Configured with strict system prompts that instruct the model to answer *only* from the provided excerpts and state when information was not found.
- **⚠️ Graceful Error Handling**: Detects and explains scanned PDFs (image-only), empty PDFs, encrypted PDFs, oversized files, and missing Ollama models.
- **💡 Suggested Questions**: Interactive one-click questions to help beginners explore their uploaded document immediately.

---

## 🏗️ Architecture & Data Flow

```
┌─────────────────┐       ┌────────────────────────┐       ┌───────────────────────┐
│                 │       │   Document Processor   │       │   ChromaDB Manager    │
│  Uploaded PDF   │ ───►  │  (PyPDF + Chunking)    │ ───►  │  (Isolated Collection │
│                 │       │  RecursiveTextSplitter │       │   nomic-embed-text)   │
└─────────────────┘       └────────────────────────┘       └───────────┬───────────┘
                                                                       │
┌─────────────────┐       ┌────────────────────────┐                   │
│   User Prompt   │ ───►  │     RAG Pipeline       │ ◄─────────────────┘
│   (Question)    │       │ (Retriever + Top-k)    │   Top-k Passages + Page Numbers
└─────────────────┘       └───────────┬────────────┘
                                      │ Context + Strict Grounding Prompt
                                      ▼
                          ┌────────────────────────┐
                          │   Local ChatOllama     │
                          │     (llama3.2:3b)      │
                          └───────────┬────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  Streamlit Interface   │
                          │ Grounded Answer +      │
                          │ Collapsible Citations  │
                          └────────────────────────┘
```

### End-to-End Data Flow:
1. **Upload**: User uploads a PDF in Streamlit (`app.py`). Raw file bytes are read in-memory.
2. **Text & Metadata Extraction**: `pypdf` extracts text page-by-page, attaching metadata (`source: filename`, `page: page_number`).
3. **Chunking**: `RecursiveCharacterTextSplitter` divides pages into overlapping chunks (800 chars, 150 overlap) while copying metadata to each chunk.
4. **Vector Embedding**: Local `nomic-embed-text` converts each text chunk into a 768-dimensional numerical vector.
5. **ChromaDB Storage**: Chunks and vectors are stored in a dedicated local collection (`./chroma_db/doc_<hash>`).
6. **Query & Retrieval**: When the user asks a question, ChromaDB compares the question vector against chunk vectors using cosine distance and fetches the top-k most relevant excerpts.
7. **Prompt Assembly**: The retrieved passages and page numbers are inserted into a strict grounding prompt template along with the user's question.
8. **Generation**: Local `llama3.2:3b` generates an answer based strictly on the excerpts.
9. **Display**: Streamlit displays the answer and an expandable card showing the filename, page number, and exact text excerpts.

---

## 📁 Project Structure & Module Roles

```
chat-with-doc/
│
├── config.py                 # Configuration constants, model names, limits & prompt templates
├── document_processor.py     # PDF validation, text extraction, page metadata & chunking
├── vector_store.py           # ChromaDB manager with isolated collections & local embeddings
├── rag_chain.py              # Retrieval logic, strict grounded prompt & Ollama inference
├── app.py                    # Streamlit web application frontend
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes venvs, Chroma database, and temporary files
├── README.md                 # Complete documentation & interview preparation
│
└── tests/                    # 18 automated regression and audit tests
    ├── conftest.py           # Pure-Python PDF synthesizers & fixtures
    ├── test_document_processor.py # PDF extraction, metadata, and error handling tests
    ├── test_vector_store.py  # Chroma collection isolation and cleanup tests
    ├── test_rag_chain.py     # Grounded answering, citations, and health check tests
    └── test_audit_qa.py      # Mocked & live tests with known facts across 3 question types
```

### Module Responsibilities:
- **[`config.py`](file:///c:/Users/pranj/OneDrive/Documents/Projects/chat-with-doc/config.py)**: Centralizes configuration constants (`DEFAULT_LLM_MODEL = "llama3.2:3b"`, `DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"`, chunk sizes, upload limits, and the grounded system prompt template).
- **[`document_processor.py`](file:///c:/Users/pranj/OneDrive/Documents/Projects/chat-with-doc/document_processor.py)**:
  - `extract_pages_from_pdf()`: Extracts text page-by-page with `pypdf`, preserving `source` and `page` metadata. Detects scanned or empty PDFs.
  - `split_documents_into_chunks()`: Splits documents using `RecursiveCharacterTextSplitter`.
  - `process_uploaded_pdf()`: End-to-end processing orchestrator.
- **[`vector_store.py`](file:///c:/Users/pranj/OneDrive/Documents/Projects/chat-with-doc/vector_store.py)**:
  - `VectorStoreManager`: Manages ChromaDB collections.
  - `index_document()`: Indexes chunks with `nomic-embed-text` into an isolated collection.
  - `search_relevant_chunks()`: Performs similarity search with distance scores.
  - `delete_collection()`: Removes collections when a document is cleared or replaced.
- **[`rag_chain.py`](file:///c:/Users/pranj/OneDrive/Documents/Projects/chat-with-doc/rag_chain.py)**:
  - `check_ollama_health()`: Verifies local Ollama server connectivity and model availability.
  - `format_context_with_citations()`: Builds structured context blocks and `Citation` objects.
  - `RAGPipeline.answer_question()`: Coordinates retrieval, prompt construction, and local `ChatOllama` generation.
- **[`app.py`](file:///c:/Users/pranj/OneDrive/Documents/Projects/chat-with-doc/app.py)**: Modern Streamlit frontend with suggested question buttons, file upload, document statistics, chat stream, and expandable citations.

---

## ❓ Why RAG? Core Conceptual Insights

### 1. Why are embeddings and vector search needed?
Standard keyword search (like `Ctrl+F` or SQL `LIKE '%word%'`) fails when the user asks a question using different words than the document (e.g., searching *"revenue growth"* when the document says *"sales surged by 15%"*). 
**Embeddings** map text into a geometric space where semantically similar sentences are close to each other. **Vector search** finds relevant passages based on conceptual meaning rather than exact keywords.

### 2. Why is retrieved context sent to the LLM?
Pre-trained LLMs only know what was in their training data up to their cutoff date; they know nothing about your private PDF. Sending the retrieved passages directly in the prompt gives the LLM the exact facts it needs to synthesize an accurate answer with citations.

---

## 🚀 Quickstart Guide (Windows PowerShell)

### 1. Prerequisites
- **Python**: Version 3.10, 3.11, or 3.12 installed.
- **Ollama**: Installed and running locally.

### 2. Install Ollama & Pull Models
1. Download Ollama for Windows from [ollama.com/download](https://ollama.com/download).
2. Open Windows PowerShell and pull both models:
   ```powershell
   # Pull local LLM (~2.0 GB)
   ollama pull llama3.2:3b

   # Pull local embedding model (~274 MB)
   ollama pull nomic-embed-text
   ```
3. Verify installed models:
   ```powershell
   ollama list
   ```

### 3. Setup Virtual Environment on Windows
In PowerShell, navigate to the project directory:
```powershell
cd c:\Users\pranj\OneDrive\Documents\Projects\chat-with-doc

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 4. Run the Application
```powershell
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Testing & Verification

The project includes **18 automated tests** covering document extraction, chunking, Chroma isolation, error handling, prompt construction, and end-to-end answering.

### Run All Tests:
```powershell
pytest -v
```

### Actual Test Results (18 of 18 Passed):
```text
tests/test_audit_qa.py::test_audit_mocked_answer_exists PASSED           [  5%]
tests/test_audit_qa.py::test_audit_mocked_answer_does_not_exist PASSED   [ 11%]
tests/test_audit_qa.py::test_audit_mocked_different_topic PASSED         [ 16%]
tests/test_audit_qa.py::test_audit_real_end_to_end_known_document PASSED [ 22%]
tests/test_document_processor.py::test_extract_pages_valid_pdf PASSED    [ 27%]
tests/test_document_processor.py::test_extract_pages_empty_bytes PASSED  [ 33%]
tests/test_document_processor.py::test_extract_pages_blank_pdf PASSED    [ 38%]
tests/test_document_processor.py::test_extract_pages_scanned_pdf PASSED  [ 44%]
tests/test_document_processor.py::test_file_size_exceeded PASSED         [ 50%]
tests/test_document_processor.py::test_split_documents_into_chunks PASSED [ 55%]
tests/test_document_processor.py::test_process_uploaded_pdf_summary PASSED [ 61%]
tests/test_rag_chain.py::test_ollama_health_check PASSED                 [ 66%]
tests/test_rag_chain.py::test_format_context_with_citations PASSED       [ 72%]
tests/test_rag_chain.py::test_empty_query_handling PASSED                [ 77%]
tests/test_rag_chain.py::test_rag_pipeline_grounded_answer_and_citations PASSED [ 83%]
tests/test_vector_store.py::test_sanitize_collection_name PASSED         [ 88%]
tests/test_vector_store.py::test_document_isolation PASSED               [ 94%]
tests/test_vector_store.py::test_delete_and_replace_collection PASSED    [100%]

============================= 18 passed in 21.34s =============================
```

---

## ⚠️ Honest Project Limitations

Every engineering project involves trade-offs. Here are the honest limitations of this architecture:

1. **Text-Based PDFs Only (No OCR)**: The app uses `pypdf` to extract digital text. Scanned paper documents or image-only PDFs without an embedded text layer cannot be read (the app detects this and warns the user).
2. **Local Hardware Dependency**: Running local models (`llama3.2:3b` and `nomic-embed-text`) relies entirely on your machine's CPU/GPU and RAM. Machines without dedicated GPUs will experience higher latency (2–8 seconds per response).
3. **Fixed Chunk Boundaries**: Information split awkwardly across chunk borders may occasionally lose nuances, even with a 150-character overlap.
4. **Top-k Context Window**: If a complex question requires synthesizing facts scattered across 20 different pages, a fixed `top_k=4` might not capture all necessary excerpts.
5. **Single Document Focus**: The app is designed for deep Q&A on one active document at a time to prevent cross-document confusion and keep isolation simple.
