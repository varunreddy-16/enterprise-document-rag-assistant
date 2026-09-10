# Enterprise Document Intelligence & RAG Assistant

A portfolio-grade Retrieval-Augmented Generation (RAG) application for natural-language Q&A over synthetic enterprise documents.

## Features
- TXT/PDF document ingestion
- Recursive chunking with overlap
- Local Sentence Transformer embeddings
- FAISS semantic search
- LangChain prompts/document abstractions
- Optional OpenAI-compatible LLM generation
- Source-aware answers
- Streamlit UI
- Retrieval evaluation
- Unit tests

> All documents in `data/documents/` are synthetic demonstration data.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
streamlit run app.py
```

Without an API key the app runs in retrieval-only mode. For generated grounded answers, copy `.env.example` to `.env` and configure an OpenAI-compatible provider.

## Evaluation

```bash
python scripts/evaluate_retrieval.py
pytest -q
```

## Example questions
- What is the password rotation requirement?
- How quickly must employees report phishing?
- Who approves a software purchase of INR 100000?
- What is the annual learning allowance?
- How many weeks of parental leave are provided?

## Architecture

Documents → extraction → chunking → embeddings → FAISS → top-k retrieval → grounded prompt → optional LLM → answer + source references.

## Limitations

This is a portfolio demonstration, not a production enterprise deployment. It does not implement authentication, document-level permissions, PII redaction, continuous indexing, or a managed vector database.
