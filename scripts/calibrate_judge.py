"""用 DuReader 人工标注校准 judge 门控阈值（Phase 3）。

思路：
- 对每个标注 query，跑"retrieve top20 → rerank top4"，
  二值真值 = 重排后 top4 是否含至少一个标注相关段（Recall>0 = 检索成功）。
- 统计"成功组"和"失败组"的 top1 rerank 分数分布，扫阈值找最佳分界
  （Youden's J = TPR - FPR 最大点）。
- 结论：RuleBasedJudge 的阈值由此定，不拍脑袋。
  这个校准过程本身可在面试里讲：门控阈值来自数据，不是玄学。

用法：
    PYTHONPATH=src python scripts/calibrate_judge.py [--limit 500]
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


def collect(retriever, reranker, queries, rel_map, limit=None):
    """返回 [(top1_score, hit_bool), ...]"""
    rows = []
    targets = queries[:limit] if limit else queries
    t0 = time.time()
    for i, q in enumerate(targets):
        relevant = rel_map.get(q["qid"])
        if not relevant:
            continue
        retrieved = retriever.retrieve(q["text"], top_k=config.TOP_K_RETRIEVE)
        ranked = reranker.rerank(q["text"], retrieved, top_k=config.TOP_K_FINAL)
        top1 = ranked[0]["rerank_score"] if ranked else -99.0
        hit = any(r["id"] in relevant for r in ranked)
        rows.append((top1, hit))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(targets)} 已处理，耗时 {time.time() - t0:.0f}s")
    return rows


def best_threshold(rows):
    """扫阈值找 Youden's J 最大点。返回 (threshold, tpr, fpr, j)。"""
    scores = sorted(r[0] for r in rows)
    lo, hi = scores[0], scores[-1]
    best = None
    steps = 60
    for i in range(steps + 1):
        t = lo + (hi - lo) * i / steps
        tp = sum(1 for s, h in rows if s >= t and h)
        fn = sum(1 for s, h in rows if s < t and h)
        fp = sum(1 for s, h in rows if s >= t and not h)
        tn = sum(1 for s, h in rows if s < t and not h)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        j = tpr - fpr
        if best is None or j > best[3]:
            best = (t, tpr, fpr, j)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="校准 judge 门控阈值")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_local()
    rel_map = defaultdict(set)
    for r in data["qrels"]:
        rel_map[r["qid"]].add(r["pid"])

    retriever = Retriever()
    reranker = get_reranker()
    rows = collect(retriever, reranker, data["queries"], rel_map, args.limit)

    hit_scores = [s for s, h in rows if h]
    miss_scores = [s for s, h in rows if not h]
    print(f"\n样本: {len(rows)}  (检索成功 {len(hit_scores)} / 失败 {len(miss_scores)})")
    if hit_scores:
        print(
            f"成功组 top1 rerank: min={min(hit_scores):.2f} "
            f"p10={sorted(hit_scores)[len(hit_scores) // 10]:.2f} "
            f"median={sorted(hit_scores)[len(hit_scores) // 2]:.2f}"
        )
    if miss_scores:
        print(
            f"失败组 top1 rerank: max={max(miss_scores):.2f} "
            f"p90={sorted(miss_scores)[len(miss_scores) * 9 // 10]:.2f} "
            f"median={sorted(miss_scores)[len(miss_scores) // 2]:.2f}"
        )

    t, tpr, fpr, j = best_threshold(rows)
    report = {
        "threshold": round(t, 3),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "youden_j": round(j, 4),
        "n": len(rows),
        "n_hit": len(hit_scores),
        "n_miss": len(miss_scores),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"\n建议 RuleBasedJudge(threshold={t:.3f})："
        f"漏判率(成功但被判不够)={1 - tpr:.1%}，"
        f"误判率(失败但被判够)={fpr:.1%}"
    )

    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = config.TRACE_DIR / "judge_calibration.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"trace 已写入: {trace_path}")


if __name__ == "__main__":
    main()
