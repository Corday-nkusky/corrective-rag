"""闭环 A/B：纠错循环 vs 朴素检索（Phase 3 收官）。

朴素 = retrieve top4 直接进生成。
闭环 = retrieve top20 → rerank top4 → judge → 不够则重写重试
     → MAX_ITERATIONS 次仍不够则 abstain。

指标（二值 qrels）：
- 检索质量：Recall@4 / hit@4（闭环取最终进入生成的那批文档）
- 行为：abstain 率、平均迭代次数、改写生效比例
- 成本：平均每 query 耗时

注意：当前 judge 为规则版（阈值 0.983 来自校准），且已实测
top1 rerank 分数区分度弱（失败组 p90 > 0.99）——规则 judge 的
门控能力有限，本 A/B 的闭环增益主要来自 rerank + abstain，
LLM judge 版待 API key 补齐后重跑，数字会更好。

用法：
    PYTHONPATH=src python scripts/eval_corrective.py [--limit 500]
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
from corrective_rag.pipeline import CorrectiveRAG  # noqa: E402
from corrective_rag.retrieve import Retriever  # noqa: E402

JUDGE_THRESHOLD = 0.983


def main() -> None:
    parser = argparse.ArgumentParser(description="纠错闭环 A/B 评测")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_local()
    rel_map = defaultdict(set)
    for r in data["qrels"]:
        rel_map[r["qid"]].add(r["pid"])

    retriever = Retriever()
    pipe = CorrectiveRAG(retriever=retriever, judge_threshold=JUDGE_THRESHOLD)
    targets = [q for q in data["queries"] if q["qid"] in rel_map]
    if args.limit:
        targets = targets[: args.limit]

    k = config.TOP_K_FINAL
    n = 0
    naive_recall = closed_recall = naive_hit = closed_hit = 0.0
    abstained = 0
    iters = 0
    rewrites_applied = 0
    t0 = time.time()

    for i, q in enumerate(targets):
        relevant = rel_map[q["qid"]]

        # 朴素基线：retrieve top4
        naive_top = {r["id"] for r in retriever.retrieve(q["text"], top_k=k)}
        naive_recall += len(naive_top & relevant) / len(relevant)
        naive_hit += 1.0 if naive_top & relevant else 0.0

        # 闭环
        r = pipe.run(q["text"])
        final_top = {c["id"] for c in r["ranked"]}
        if r["status"] == "answered":
            closed_recall += len(final_top & relevant) / len(relevant)
            closed_hit += 1.0 if final_top & relevant else 0.0
        else:
            abstained += 1
        iters += r["iteration"]
        rewrites_applied += sum(
            1 for t in r["trace"] if t["step"] == "rewrite" and t.get("applied")
        )
        n += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(targets)}，耗时 {time.time() - t0:.0f}s")

    elapsed = time.time() - t0
    answered = n - abstained
    report = {
        "n_queries": n,
        "k": k,
        "judge": "rule-based(threshold=0.983)",
        "naive": {
            "recall@4": round(naive_recall / n, 4),
            "hit@4": round(naive_hit / n, 4),
        },
        "closed_loop": {
            "recall@4(answered only)": round(closed_recall / max(1, answered), 4),
            "hit@4(answered only)": round(closed_hit / max(1, answered), 4),
            "abstain_rate": round(abstained / n, 4),
            "avg_iterations": round(iters / n, 2),
            "rewrite_applied_per_query": round(rewrites_applied / n, 2),
        },
        "elapsed_sec": round(elapsed, 1),
        "sec_per_query": round(elapsed / n, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = (
        config.TRACE_DIR
        / f"corrective_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"trace 已写入: {trace_path}")


if __name__ == "__main__":
    main()
