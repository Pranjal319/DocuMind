# 🎓 DocuMind: RAG & GenAI Interview Preparation Guide

This guide contains clear, beginner-friendly answers to common interview questions based on the actual code and architecture in the **DocuMind** repository.

---

### 1. What is RAG?
> **Answer**:  
> RAG stands for **Retrieval-Augmented Generation**. Standard LLMs only know facts from their pre-training data up to their knowledge cutoff date and have no access to private, company-specific documents. RAG solves this in three steps:
> 1. **Retrieve**: When a user asks a question, we search a local database for the most relevant excerpts from our private documents.
> 2. **Augment**: We insert those retrieved excerpts into the prompt alongside the question.
> 3. **Generate**: We pass the augmented prompt to an LLM, which writes an answer based strictly on those excerpts with page citations.

---

### 2. Why did you build this project?
> **Answer**:  
> "I built this project to gain hands-on, practical experience with the core building blocks of GenAI and LangChain: document loading, chunking, vector embeddings, vector databases, and prompt grounding.  
> I also wanted to ensure the application is **100% private, free, and local** using Ollama and ChromaDB, so sensitive documents never leave the computer and no paid API keys or cloud services are required."

---

### 3. What is LangChain?
> **Answer**:  
> LangChain is an open-source framework designed to simplify building applications with Large Language Models. Instead of writing custom HTTP requests and formatting logic from scratch, LangChain provides standardized components for:
> - Data ingestion (`Document`)
> - Text splitting (`RecursiveCharacterTextSplitter`)
> - Vector stores (`Chroma`)
> - Embeddings (`OllamaEmbeddings`)
> - Chat models (`ChatOllama`)
> - Prompt templates (`PromptTemplate`)

---

### 4. What is a Document object?
> **Answer**:  
> A `Document` is LangChain's standardized data container. It has two main attributes:
> - `page_content` (`str`): The raw text string.
> - `metadata` (`dict`): A dictionary storing context about that text (e.g., `{"source": "report.pdf", "page": 4}`).  
> In our project, metadata is essential because it allows us to show exact page citations beneath every answer.

---

### 5. What are embeddings?
> **Answer**:  
> An embedding model (like `nomic-embed-text`) takes a piece of text and converts it into a list of floating-point numbers called a **vector** (in our case, 768 numbers).  
> These numbers represent the semantic meaning of the text. Sentences with similar meanings (e.g., *"revenue increased"* and *"sales went up"*) will have vectors that point in almost the same direction in multi-dimensional space, even if they use completely different words.

---

### 6. What is ChromaDB?
> **Answer**:  
> ChromaDB is an open-source, lightweight vector database designed to store and query high-dimensional embeddings.  
> Unlike traditional relational databases that search for exact text matches with SQL, ChromaDB indexes vectors using algorithms like HNSW (Hierarchical Navigable Small World) to perform fast nearest-neighbor similarity searches. In our project, it runs locally and saves vectors to disk in `./chroma_db/`.

---

### 7. What is similarity search?
> **Answer**:  
> When a user enters a question, our embedding model turns that question into a vector.  
> ChromaDB calculates the mathematical distance (specifically **Cosine Distance**) between the question vector and every stored chunk vector. The chunks with the smallest distance (highest similarity) are returned as the most relevant passages.

---

### 8. What are chunk size and chunk overlap?
> **Answer**:  
> - **Chunk size (e.g., 800 characters)**: The maximum character length of any single piece of text. We chunk because LLMs have context limits and retrieve more accurately from focused paragraphs than entire books.
> - **Chunk overlap (e.g., 150 characters)**: The number of characters shared between adjacent chunks. Overlap ensures that a sentence sitting right on a chunk boundary isn't sliced in half, preserving context across chunk transitions.

---

### 9. What is the difference between retrieval and generation?
> **Answer**:  
> - **Retrieval** is finding the right information. It is handled by the embedding model and vector database (ChromaDB). It searches through the document chunks and fetches the top relevant excerpts.
> - **Generation** is writing the answer. It is handled by the LLM (`llama3.2:3b`). It reads the retrieved excerpts and synthesizes a coherent, human-readable answer that addresses the user's question.

---

### 10. How does your application reduce hallucinations?
> **Answer**:  
> Our application uses three complementary grounding strategies:
> 1. **Strict Prompt Grounding**: The system prompt explicitly commands the model: *"Rely ONLY on the information present in the excerpts... Do NOT use outside knowledge... If the excerpts do not contain enough facts, reply: 'I could not find the answer in the provided document context.'"*
> 2. **Low Temperature (0.1)**: Setting the LLM temperature low makes the output deterministic and factual rather than creative.
> 3. **Verifiable Citations**: Every answer displays the exact source filename, page number, and retrieved excerpts so the user can immediately verify the claim against the original PDF.

---

### 11. Why did you use Ollama?
> **Answer**:  
> Ollama makes it simple to run open-weight models (like `llama3.2:3b` and `nomic-embed-text`) locally on Windows, macOS, and Linux without complex Docker setups or Python C++ bindings.  
> It provides a clean local REST API (`http://127.0.0.1:11434`), requires zero API keys, incurs no costs, and ensures complete data privacy because no data is sent across the internet.

---

### 12. What are the limitations of your project?
> **Answer**:  
> "I designed this project with clean trade-offs:
> 1. **Text-based PDFs only**: We use `pypdf`, which reads digital text. It does not include an OCR engine like Tesseract, so scanned images of documents cannot be read (though our code detects this and warns the user).
> 2. **Local Hardware Constraints**: Inference speed depends on the user's computer. On CPU-only machines, generation can take 4–8 seconds.
> 3. **Fixed Top-k**: We retrieve the top 4 chunks by default. If a question requires synthesizing facts scattered across 15 different pages, a fixed top-k might miss some context.
> 4. **Single-document focus**: Each session isolates queries to one active PDF to avoid cross-document confusion, rather than indexing hundreds of cross-referenced documents at once."

---

## 🎯 Tip for Explaining the Project in an Interview
When asked to describe your project, use this concise 60-second summary:

> *"I built a 100% local, privacy-first RAG web application using Python, LangChain, Streamlit, ChromaDB, and Ollama.  
> Users upload a PDF, which is extracted with PyPDF, tagged with page metadata, and split into overlapping chunks with RecursiveCharacterTextSplitter.  
> Chunks are embedded using `nomic-embed-text` into an isolated ChromaDB collection.  
> When the user asks a question, ChromaDB retrieves the top-4 most relevant passages, which are injected into a strict grounding prompt sent to `llama3.2:3b`.  
> The UI displays the grounded answer alongside expandable citations showing the source filename, page number, and original excerpt."*
