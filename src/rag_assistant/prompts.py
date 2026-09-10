from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer ONLY from the supplied context. Do not invent policies, dates, thresholds, or procedures.
If the context does not contain enough evidence, say:
\"I don't have enough information in the indexed documents to answer that.\"
Mention the source document when useful.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

def format_context(results):
    blocks = []
    for doc, score in results:
        blocks.append(
            f"[Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')} | Score: {score:.4f}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)
