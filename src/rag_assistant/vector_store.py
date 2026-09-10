from pathlib import Path
from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings

def build_vector_store(chunks):
    return FAISS.from_documents(chunks, get_embeddings())

def save_vector_store(store, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    store.save_local(str(directory))

def load_vector_store(directory: Path):
    if not (directory / "index.faiss").exists():
        return None
    return FAISS.load_local(str(directory), get_embeddings(), allow_dangerous_deserialization=True)

def retrieve(store, query, k=4):
    return store.similarity_search_with_score(query, k=k)
