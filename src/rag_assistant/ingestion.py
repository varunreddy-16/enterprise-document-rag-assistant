from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from .config import DOCUMENT_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents(directory: Path = DOCUMENT_DIR) -> List[Document]:
    docs = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() == ".txt":
            docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name, "page": 1}))
        elif path.suffix.lower() == ".pdf":
            for page_number, page in enumerate(PdfReader(str(path)).pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": path.name, "page": page_number}))
    return docs

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
