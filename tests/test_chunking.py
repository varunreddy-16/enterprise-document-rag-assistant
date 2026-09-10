from langchain_core.documents import Document
from src.rag_assistant.ingestion import chunk_documents

def test_chunking_preserves_metadata():
    docs = [Document(page_content="A " * 500, metadata={"source":"demo.txt","page":1})]
    chunks = chunk_documents(docs)
    assert chunks
    assert all(c.metadata["source"] == "demo.txt" for c in chunks)
