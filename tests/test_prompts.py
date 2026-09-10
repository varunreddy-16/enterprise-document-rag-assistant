from langchain_core.documents import Document
from src.rag_assistant.prompts import format_context

def test_context_contains_metadata():
    result = [(Document(page_content="Passwords change every 90 days.", metadata={"source":"security.txt","page":1}), 0.1)]
    context = format_context(result)
    assert "security.txt" in context
    assert "90 days" in context
