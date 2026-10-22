"""Offline evaluation over a golden set. Run from backend/:  python -m eval.run_eval

Metrics
  retrieval hit@k : was the expected source document among the retrieved chunks?
  answer correct  : does the answer contain the expected fact(s)?   (a cheap faithfulness proxy)
  grounded        : did every RAG answer come with at least one citation?

In a real project add RAGAS / DeepEval (faithfulness, answer relevancy, context precision with an
LLM judge) and run this in CI so a prompt, chunking or model change can't silently regress quality.
"""
import asyncio
import json
from pathlib import Path

from app.agent import graph
from app.vectorstore import ingest_dir, retrieve


async def main() -> None:
    ingest_dir()
    rows = [json.loads(l) for l in Path(__file__).with_name("golden.jsonl").read_text().splitlines() if l.strip()]
    hits = correct = grounded = rag = 0
    for r in rows:
        sources = {d.metadata["source"] for d, _ in retrieve(r["question"])}
        hit = r["source"] is None or r["source"] in sources
        state = await graph.ainvoke({"question": r["question"], "trace": []})
        ok = all(m.lower() in state["answer"].lower() for m in r["must_contain"])
        if state.get("route") == "generate":
            rag += 1
            grounded += "[" in state["answer"]
        hits += hit
        correct += ok
        print(f"{'PASS' if ok else 'FAIL'}  hit={'y' if hit else 'n'}  {r['question']}\n      -> {state['answer'][:110]}")
    n = len(rows)
    print(f"\nretrieval hit@k {hits}/{n} = {hits / n:.0%}   answer correct {correct}/{n} = {correct / n:.0%}"
          f"   grounded {grounded}/{rag}")


if __name__ == "__main__":
    asyncio.run(main())
