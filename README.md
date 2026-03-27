# RAG Example 2: Company Policy Assistant

A robust Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS**, and **Ollama**. This project allows you to ingest PDF documents (like company policies) and ask questions about them using local LLMs.

---

## 🚀 Features

- **Robust Ingestion**: Uses `PyPDFLoader` and `RecursiveCharacterTextSplitter` for clean and efficient data processing.
- **Unified Embeddings**: Uses `BAAI/bge-small-en` via `langchain_huggingface` for high-quality, consistent semantic search.
- **Local LLM Integration**: Powered by **Ollama** (`llama3`), ensuring your data stays private and runs locally.
- **Portable Architecture**: Scripts use relative paths for easy setup and deployment.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** (installed and running with `llama3` pulled: `ollama pull llama3`)

### 2. Environment Setup
Clone the repository and create a virtual environment:
```bash
git clone https://github.com/ramakrishna0028/rag_example_2.git
cd rag_example_2
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r req.txt
```

---

## 📖 Usage Guide

### Step 1: Ingest Data
Place your PDF files in the `data/` directory (the default is `sample_company_policies_rag.pdf`). Then, run the ingestion script to build the vector index:
```bash
python ingestion.py
```
This will create a `collection/` folder containing your searchable index.

### Step 2: Query the Assistant
Once the index is created, you can ask questions by modifying the `query` variable in `query.py` or just running the script:
```bash
python query.py
```

---

## 🏗️ Project Structure
```text
.
├── data/               # Source PDF documents
├── collection/         # Generated FAISS vector index (ignored by git)
├── ingestion.py        # PDF processing and indexing script
├── query.py            # RAG query and LLM generation script
├── req.txt             # Project dependencies
└── .gitignore          # Git exclusion rules
```

---

## 📝 Example Output
**Query**: *"what is status of policies no 6"*  
**Response**: *"Policy #6 is the Expense Reimbursement Policy. According to this policy, business-related expenses may be reimbursed..."*

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
