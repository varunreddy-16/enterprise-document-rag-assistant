from langchain_core.documents import Document
from src.rag_assistant.vector_store import build_vector_store, retrieve

def test_retrieval_returns_k():
    docs = [
        Document(page_content="Passwords change every 90 days.", metadata={"source":"security.txt"}),
        Document(page_content="Annual leave is 18 days.", metadata={"source":"handbook.txt"}),
        Document(page_content="Learning allowance is INR 40000.", metadata={"source":"benefits.txt"})
    ]
    store = build_vector_store(docs)
    assert len(retrieve(store, "password rotation", k=2)) == 2
