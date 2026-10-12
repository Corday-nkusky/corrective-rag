"""闭环 A/B：纠错循环 vs 朴素检索（Phase 3 收官）。

朴素 = retrieve top4 直接进生成。
闭环 = retrieve top20 → rerank top4 → judge → 不够则重写重试
     → MAX_ITERATIONS 次仍不够则 abstain。

指标（二值 qrels）：
- 检索质量：Recall@4 / hit@4（闭环取最终进入生成的那批文档）
- 行为：abstain 率、平均迭代次数、改写生效比例
- 成本：平均每 query 耗时

注意：judge 自动选择——有 LLM_API_KEY 用 LLM judge，无 key 回落规则版
（阈值 0.983 来自校准）。规则版已实测弱（top1 rerank 分数饱和无区分度），
LLM judge 版为正式口径。

用法：
    PYTHONPATH=src python scripts/eval_corrective.py [--limit 500]

断点续跑：逐条结果追加写入 data/traces/ab_progress_<run_id>.jsonl，
中途被杀再重跑同一 run_id 可跳过已完成条目（防 API 费用白花）。
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
from corrective_rag.cost import tracker  # noqa: E402
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
    t0 = time.time()

    # 逐条落盘（flush）：中途被杀也保留已花钱的结果，重跑同 run_id 可跳过
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = config.TRACE_DIR / f"ab_progress_{run_id}.jsonl"
    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    done_qids = set()
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_qids.add(json.loads(line)["qid"])
    progress_f = open(progress_path, "a", encoding="utf-8")

    for i, q in enumerate(targets):
        if q["qid"] in done_qids:
            continue
        relevant = rel_map[q["qid"]]

        # 朴素基线：retrieve top4
        naive_top = {r["id"] for r in retriever.retrieve(q["text"], top_k=k)}

        # 闭环
        r = pipe.run(q["text"])
        final_top = {c["id"] for c in r["ranked"]}
        status = r["status"]
        n += 1
        progress_f.write(
            json.dumps(
                {
                    "qid": q["qid"],
                    "query": q["text"],
                    "status": status,
                    "iteration": r["iteration"],
                    "naive_hit": 1.0 if naive_top & relevant else 0.0,
                    "closed_hit": q_hit,
                    "naive_recall": len(naive_top & relevant) / len(relevant),
                    "closed_recall": q_recall,
                    "judge_levels": [
                        t.get("level") for t in r["trace"] if t["step"] == "judge"
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        progress_f.flush()
        print(f"  {i + 1}/{len(targets)}，耗时 {time.time() - t0:.0f}s", flush=True)

    progress_f.close()
    elapsed = time.time() - t0
    from corrective_rag.judge import LLMJudge

    judge_desc = (
        f"llm({config.LLM_MODEL})"
        if isinstance(pipe.judge, LLMJudge)
        else f"rule-based(threshold={JUDGE_THRESHOLD})"
    )
    # 从 jsonl 全量重算：断点续跑跳过的条目也在文件里，聚合天然正确
    rows = []
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n = len(rows)
    answered_rows = [r for r in rows if r["status"] == "answered"]
    answered = len(answered_rows)
    report = {
        "n_queries": n,
        "k": k,
        "judge": judge_desc,
        "naive": {
            "recall@4": round(sum(r["naive_recall"] for r in rows) / max(1, n), 4),
            "hit@4": round(sum(r["naive_hit"] for r in rows) / max(1, n), 4),
        },
        "closed_loop": {
            "recall@4(answered only)": round(
                sum(r["closed_recall"] for r in answered_rows) / max(1, answered), 4
            ),
            "hit@4(answered only)": round(
                sum(r["closed_hit"] for r in answered_rows) / max(1, answered), 4
            ),
            "abstain_rate": round((n - answered) / max(1, n), 4),
            "avg_iterations": round(sum(r["iteration"] for r in rows) / max(1, n), 2),
        },
        "cost": {
            **tracker.snapshot(),
            "cny_per_query": round(tracker.cost_cny(config.LLM_MODEL) / max(1, n), 5),
        },
        "elapsed_sec": round(elapsed, 1),
        "sec_per_query": round(elapsed / max(1, n), 2),
        "progress_file": str(progress_path),
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
