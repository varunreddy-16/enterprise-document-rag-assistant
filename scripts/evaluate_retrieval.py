import json
from src.rag_assistant.config import EVAL_FILE, TOP_K
from src.rag_assistant.rag_pipeline import get_or_build_store
from src.rag_assistant.vector_store import retrieve

cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
hits = 0
for case in cases:
    results = retrieve(get_or_build_store(), case["question"], TOP_K)
    sources = {doc.metadata.get("source") for doc, _ in results}
    ok = case["expected_source"] in sources
    hits += int(ok)
    print(("PASS" if ok else "FAIL"), case["question"])
print(f"Recall@{TOP_K}: {hits/len(cases):.2%}")
