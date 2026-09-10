import argparse
from langchain_openai import ChatOpenAI
from .config import INDEX_DIR, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL, TOP_K
from .ingestion import load_documents, chunk_documents
from .vector_store import load_vector_store, build_vector_store, save_vector_store, retrieve
from .prompts import PROMPT, format_context

def get_or_build_store():
    store = load_vector_store(INDEX_DIR)
    if store:
        return store
    store = build_vector_store(chunk_documents(load_documents()))
    save_vector_store(store, INDEX_DIR)
    return store

def make_llm():
    if not OPENAI_API_KEY:
        return None
    kwargs = {"model": OPENAI_MODEL, "temperature": 0}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(api_key=OPENAI_API_KEY, **kwargs)

def answer_question(question, top_k=TOP_K):
    results = retrieve(get_or_build_store(), question, k=top_k)
    context = format_context(results)
    llm = make_llm()
    if llm:
        response = llm.invoke(PROMPT.format_messages(question=question, context=context))
        answer = response.content
    else:
        answer = "LLM generation is disabled. Retrieved evidence:\n\n" + context
    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
                "score": float(s), "chunk_id": d.metadata.get("chunk_id")}
               for d, s in results]
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    result = answer_question(args.question, args.top_k)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(source)
