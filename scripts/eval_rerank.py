"""reranker 增益评测：朴素 top4 vs 检索top20+重排top4（Phase 3）。

回答的问题：cross-encoder 重排到底把多少 query 的相关段落顶进了 top4。
这是 retrieve-and-rerank 两段式的 A/B，简历数字的直接出处。

口径（二值 qrels，与 eval_retrieval.py 一致）：
- Recall@4 = |top4 ∩ relevant| / |relevant|
- hit@4   = top4 是否命中至少一个 relevant

用法：
    PYTHONPATH=src python scripts/eval_rerank.py [--limit 500]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.data_loader import load_local  # noqa: E402
from corrective_rag.rerank import get_reranker  # noqa: E402
from corrective_rag.retrieve import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="reranker 增益 A/B 评测")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_local()
    rel_map = defaultdict(set)
    for r in data["qrels"]:
        rel_map[r["qid"]].add(r["pid"])

    retriever = Retriever()
    reranker = get_reranker()
    targets = [q for q in data["queries"] if q["qid"] in rel_map]
    if args.limit:
        targets = targets[: args.limit]

    k = config.TOP_K_FINAL
    n = 0
    naive_recall = naive_hit = rerank_recall = rerank_hit = 0.0
    rescued = 0
    broken = 0
    t0 = time.time()

    for i, q in enumerate(targets):
        relevant = rel_map[q["qid"]]
        retrieved = retriever.retrieve(q["text"], top_k=config.TOP_K_RETRIEVE)
        ranked = reranker.rerank(q["text"], retrieved, top_k=k)

        naive_top = {r["id"] for r in retrieved[:k]}
        rerank_top = {r["id"] for r in ranked}

        naive_recall += len(naive_top & relevant) / len(relevant)
        naive_hit += 1.0 if naive_top & relevant else 0.0
        rerank_recall += len(rerank_top & relevant) / len(relevant)
        rerank_hit += 1.0 if rerank_top & relevant else 0.0

        if rerank_top & relevant and not naive_top & relevant:
            rescued += 1
        if naive_top & relevant and not rerank_top & relevant:
            broken += 1
        n += 1
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(targets)}，耗时 {time.time() - t0:.0f}s")

    report = {
        "n_queries": n,
        "k": k,
        "retrieve_top_k": config.TOP_K_RETRIEVE,
        "naive": {
            "recall@4": round(naive_recall / n, 4),
            "hit@4": round(naive_hit / n, 4),
        },
        "reranked": {
            "recall@4": round(rerank_recall / n, 4),
            "hit@4": round(rerank_hit / n, 4),
        },
        "rescued_by_rerank": rescued,
        "broken_by_rerank": broken,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = (
        config.TRACE_DIR
        / f"rerank_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"trace 已写入: {trace_path}")


if __name__ == "__main__":
    main()
