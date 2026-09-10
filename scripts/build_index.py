from src.rag_assistant.config import INDEX_DIR
from src.rag_assistant.ingestion import load_documents, chunk_documents
from src.rag_assistant.vector_store import build_vector_store, save_vector_store

documents = load_documents()
chunks = chunk_documents(documents)
save_vector_store(build_vector_store(chunks), INDEX_DIR)
print(f"Loaded {len(documents)} documents and indexed {len(chunks)} chunks.")
