# 🦙 LLaMA 2 Local AI Project (via Ollama)

This project runs a local LLaMA 2 model using [Ollama](https://ollama.com) for natural language tasks such as chat, summarization, or Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- ✅ Local inference using Meta's LLaMA 2 model
- ✅ Powered by [Ollama](https://ollama.com) for easy model management
- ✅ Integrates with vector stores (e.g., Chroma) for RAG pipelines
- ✅ Lightweight and private — runs entirely on your machine

---

## 🛠️ Prerequisites

- [Ollama](https://ollama.com/download) installed
- Python 3.8+
- `pip` for Python package management

---

## 📦 Setup

1. **Install Python dependencies**

```bash
pip install -r requirements.txt

ollama pull llama2

python .\main.py

-------------------------------
Ask your question (q to quit): Do you have gluten-free pizza?