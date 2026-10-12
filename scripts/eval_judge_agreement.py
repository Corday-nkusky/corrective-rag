"""LLM judge 与 DuReader 人工标注的一致率评测（Phase 3）。

口径（严格版，面试可展开）：
- 每个 query 跑 retrieve top20 → rerank top4 → LLM judge（level>=1 记 sufficient）
- 人工二值真值 = 重排后 top4 是否含至少一个标注相关段（hit）

⚠️ 边界必须主动披露：DuReader 的 qrels 只标"相关"、不标"充分"，
所以真值是"检索命中"而非"资料充分回答问题"——这是 judge 门控有效性的
代理指标，不是充分性本身的真值。这么说比装不知道高一个段位。

混淆矩阵四格（judge 判断 × 人工真值）：
- TP = judge够 & hit    → 正确放行
- TN = judge不够 & miss → 正确拦截
- FN = judge不够 & hit  → 漏判（资料其实命中，被拦下重试/拒答，白付代价）
- FP = judge够 & miss   → 误判（资料没中还放行，幻觉直接漏出去）

指标：agreement=(TP+TN)/N；precision=TP/(TP+FP)（判"够"的可信度）；
miss_rate=FN/(TP+FN)（把能答的拦掉的概率）。

用法：
    PYTHONPATH=src python scripts/eval_judge_agreement.py [--limit 300]
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
from corrective_rag.judge import LLMJudge, RuleBasedJudge  # noqa: E402
from corrective_rag.rerank import get_reranker  # noqa: E402
from corrective_rag.retrieve import Retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge 与人工标注一致率")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--judge",
        choices=["auto", "llm", "rule"],
        default="auto",
        help="auto=有 key 用 LLM，无 key 用规则（对照跑）",
    )
    args = parser.parse_args()

    data = load_local()
    rel_map = defaultdict(set)
    for r in data["qrels"]:
        rel_map[r["qid"]].add(r["pid"])

    retriever = Retriever()
    reranker = get_reranker()
    if args.judge == "llm" or (args.judge == "auto" and config.LLM_API_KEY):
        judge = LLMJudge()
        judge_desc = f"llm({config.LLM_MODEL})"
    else:
        judge = RuleBasedJudge(threshold=0.983)
        judge_desc = "rule(threshold=0.983)"

    targets = [q for q in data["queries"] if q["qid"] in rel_map]
    if args.limit:
        targets = targets[: args.limit]

    tp = tn = fn = fp = 0
    levels = defaultdict(int)
    t0 = time.time()

    # 逐条落盘（flush）：中途被杀也保留已花钱的结果
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = config.TRACE_DIR / f"judge_agreement_progress_{run_id}.jsonl"
    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    progress_f = open(progress_path, "a", encoding="utf-8")

    for i, q in enumerate(targets):
        relevant = rel_map[q["qid"]]
        retrieved = retriever.retrieve(q["text"], top_k=config.TOP_K_RETRIEVE)
        ranked = reranker.rerank(q["text"], retrieved, top_k=config.TOP_K_FINAL)
        hit = any(r["id"] in relevant for r in ranked)

        verdict = judge.judge(q["text"], ranked)
        sufficient = verdict.get("sufficient", False)
        levels[verdict.get("level", -1)] += 1

        if sufficient and hit:
            tp += 1
            cell = "TP"
        elif sufficient and not hit:
            fp += 1
            cell = "FP"
        elif not sufficient and hit:
            fn += 1
            cell = "FN"
        else:
            tn += 1
            cell = "TN"
        progress_f.write(
            json.dumps(
                {
                    "qid": q["qid"],
                    "query": q["text"],
                    "hit": hit,
                    "level": verdict.get("level"),
                    "sufficient": sufficient,
                    "cell": cell,
                    "reason": verdict.get("reason", ""),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        progress_f.flush()
        print(
            f"  {i + 1}/{len(targets)}，耗时 {time.time() - t0:.0f}s，"
            f"累计花费 ¥{tracker.cost_cny(config.LLM_MODEL):.3f}",
            flush=True,
        )

    progress_f.close()
    n = len(targets)
    report = {
        "judge": judge_desc,
        "n": n,
        "confusion": {
            "TP_放行且命中": tp,
            "TN_拦截且未命中": tn,
            "FN_漏判_拦掉了能答的": fn,
            "FP_误判_放行了没资料的": fp,
        },
        "agreement": round((tp + tn) / max(1, n), 4),
        "precision_判够的可信度": round(tp / max(1, tp + fp), 4),
        "miss_rate_能答被拦率": round(fn / max(1, tp + fn), 4),
        "level_distribution": dict(levels),
        "progress_file": str(progress_path),
        "cost": {
            **tracker.snapshot(),
            "cny_per_query": round(tracker.cost_cny(config.LLM_MODEL) / max(1, n), 5),
        },
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = (
        config.TRACE_DIR
        / f"judge_agreement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"trace 已写入: {trace_path}")


if __name__ == "__main__":
    main()
