"""朴素检索基线评测：DuReader-retrieval dev split，2,000 条人工标注 query（Phase 2）。

指标（二值 qrels 口径）：
- Recall@k = |top-k ∩ relevant| / |relevant|，按 query 平均
- hit@k   = top-k 是否命中至少一个 relevant（0/1），按 query 平均
- MRR@10  = 1 / 第一个 relevant 的排名（排名 > 10 记 0），按 query 平均
  与 DuReader 论文主指标口径一致（论文基线：BM25 MRR@10=21.03，
  双编码器=53.96，双编码器+cross-encoder=74.21），可直接对照。

用法：
    PYTHONPATH=src python scripts/eval_retrieval.py [--limit 200] [--k 1 5 10 20 50]
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
from corrective_rag.retrieve import Retriever  # noqa: E402


def evaluate(retriever: Retriever, queries, qrels, k_list, limit=None):
    rel_map = defaultdict(set)
    for r in qrels:
        rel_map[r["qid"]].add(r["pid"])

    max_k = max(k_list)
    targets = queries[:limit] if limit else queries

    agg = {f"recall@{k}": 0.0 for k in k_list}
    agg.update({f"hit@{k}": 0.0 for k in k_list})
    agg["mrr@10"] = 0.0
    n = 0
    t0 = time.time()

    for q in targets:
        qid = q["qid"]
        relevant = rel_map.get(qid)
        if not relevant:
            continue  # 无标注 query 不参与（data_loader 已过滤，双保险）
        results = retriever.retrieve(q["text"], top_k=max_k)
        top_pids = [r["id"] for r in results]

        for k in k_list:
            hit_set = set(top_pids[:k])
            agg[f"recall@{k}"] += len(hit_set & relevant) / len(relevant)
            agg[f"hit@{k}"] += 1.0 if hit_set & relevant else 0.0

        first_rank = next(
            (i + 1 for i, p in enumerate(top_pids[:10]) if p in relevant), None
        )
        agg["mrr@10"] += (1.0 / first_rank) if first_rank else 0.0
        n += 1

    elapsed = time.time() - t0
    out = {key: round(val / max(1, n), 4) for key, val in agg.items()}
    out["n_queries"] = n
    out["elapsed_sec"] = round(elapsed, 1)
    out["qps"] = round(n / max(elapsed, 1e-9), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="DuReader-retrieval 朴素检索基线")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    args = parser.parse_args()

    data = load_local()
    retriever = Retriever()
    print(f"索引 chunk 数: {len(retriever.chunks)}")

    report = evaluate(
        retriever, data["queries"], data["qrels"], args.k, limit=args.limit
    )
    report["timestamp"] = datetime.now(timezone.utc).isoformat()
    report["config"] = {
        "embedding_model": config.EMBEDDING_MODEL,
        "split": "dev",
        "limit": args.limit,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = (
        config.TRACE_DIR
        / f"retrieval_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"trace 已写入: {trace_path}")


if __name__ == "__main__":
    main()
