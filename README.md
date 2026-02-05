# Multi-Model RAG Pipeline with LLM-as-a-Judge Evaluation

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that demonstrates advanced AI engineering concepts, including 4-bit quantization, vector databases, and automated evaluation using a "Judge" model.

## 🚀 Overview
This project goes beyond a simple chatbot. It implements a full RAG cycle:
1.  **Ingestion:** Processes the SQuAD dataset into a high-performance vector store.
2.  **Retrieval:** Uses semantic search to find relevant context in milliseconds.
3.  **Generation:** Leverages **Llama-3.1 (8B)** to synthesize answers based strictly on retrieved data.
4.  **Evaluation:** Employs **Llama-3.2 (3B)** as an impartial judge to score responses based on faithfulness and accuracy.

## 🛠️ Tech Stack
* **Orchestration:** LangChain
* **Models:** Meta Llama-3.1-8B-Instruct & Llama-3.2-3B-Instruct
* **Quantization:** BitsAndBytes (4-bit NF4)
* **Vector Store:** ChromaDB
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Data:** SQuAD (Stanford Question Answering Dataset)

## 🏗️ Architecture
- **Chunking Strategy:** RecursiveCharacterTextSplitter (1000 char size, 100 char overlap).
- **Retrieval:** K=3 Vector search via Chroma.
- **Quantization:** `bnb_4bit_compute_dtype=torch.bfloat16` for memory efficiency.
- **Evaluation Loop:** LLM-as-a-Judge architecture that generates a numeric score (0-10) and qualitative reasoning for every output.
