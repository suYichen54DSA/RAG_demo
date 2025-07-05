# RAG_demo
Retrieval-Augmented Generation (RAG) with DeepSeek-R1 and Local Embeddings
# Retrieval-Augmented Generation (RAG) with DeepSeek-R1 and Local Embeddings

This repository demonstrates how to build a simple RAG pipeline that combines:

- **DeepSeek‑R1** for answer generation (via OpenRouter)  
- **Local HuggingFace embeddings** (`all‑MiniLM‑L6‑v2`) for vector retrieval  
- **FAISS** as the in‑memory vector store  
- **LangChain** for orchestration  

---

## 📦 Requirements

- Python 3.8 or newer  
- `langchain`  
- `langchain‑openai`  
- `langchain‑community`  
- `faiss‑cpu`  
- `sentence‑transformers`  

Install all dependencies with:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu sentence-transformers
