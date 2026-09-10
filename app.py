import streamlit as st
from src.rag_assistant.config import TOP_K
from src.rag_assistant.rag_pipeline import answer_question

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Enterprise Document Intelligence & RAG Assistant")
st.caption("Semantic retrieval over synthetic enterprise policies with optional grounded LLM generation.")

top_k = st.sidebar.slider("Retrieved chunks", 1, 8, TOP_K)
question = st.text_input("Ask a question", placeholder="What is the password rotation requirement?")

examples = [
    "What is the password rotation requirement?",
    "How quickly must employees report phishing?",
    "Who approves a software purchase of INR 100000?",
    "What is the annual learning allowance?",
    "How many weeks of parental leave are provided?",
]

if st.button("Search", type="primary") and question.strip():
    result = answer_question(question.strip(), top_k)
    st.subheader("Answer")
    st.write(result["answer"])
    st.subheader("Sources")
    for s in result["sources"]:
        st.write(f"- `{s['source']}` · page {s['page']} · score `{s['score']:.4f}`")

st.subheader("Try an example")
for example in examples:
    if st.button(example):
        result = answer_question(example, top_k)
        st.write(result["answer"])
        st.write(result["sources"])
