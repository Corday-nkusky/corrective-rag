"""闭环 A/B：纠错循环 vs 朴素检索（Phase 3 收官；Phase 7 扩到 1,000 条终测）。

朴素 = retrieve top4 直接进生成。
闭环 = retrieve top20 → rerank top4 → judge → 不够则重写重试
     → MAX_ITERATIONS 次仍不够则 abstain。

指标（二值 qrels）：
- 检索质量：Recall@4 / hit@4（闭环取最终进入生成的那批文档）
- 行为：abstain 率、平均迭代次数
- 成本：平均每 query 耗时与花费

注意：judge 自动选择——有 LLM_API_KEY 用 LLM judge，无 key 回落规则版
（阈值 0.983 来自校准）。规则版已实测弱（top1 rerank 分数饱和无区分度），
LLM judge 版为正式口径。

用法：
    PYTHONPATH=src python scripts/eval_corrective.py --limit 1000 --tag final

断点续跑：逐条结果追加写入 data/traces/ab_progress_<tag>.jsonl（**固定文件名**，
用同一 --tag 重跑会跳过已完成 qid，防 API 费用白花）。汇总阶段从进度文件
全量重算，与"本轮新跑了几条"无关。

并发：--workers 线程池。GPU 前向在 embedding/rerank 模块内加锁串行化，
网络端 LLM 调用并发——这才是把 21s/query 压下来的关键。
预算：--budget 秒，超预算当前块跑完即停，下次同 --tag 续跑。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.cost import tracker  # noqa: E402
from corrective_rag.data_loader import load_local  # noqa: E402
from corrective_rag.pipeline import CorrectiveRAG  # noqa: E402
from corrective_rag.retrieve import Retriever  # noqa: E402

JUDGE_THRESHOLD = 0.983
MAX_RETRY = 3


class _ForceRetrieveRouter:
    """强制走检索路径的桩 router（A/B 默认用它）。

    为什么必须强制：DuReader 是**检索基准**，A/B 要量的是"闭环纠错 vs
    单趟检索"在**检索轴**上的增益。router（Phase 6b）会把部分常识类 query
    判成 direct 直答 → 该 query 根本没检索 → closed_recall=0、judge 不触发，
    数字就不是同类比较（2026-10-13 冒烟实测：8 条里 2 条被判 direct，
    closed_recall 从 1.0 掉到 0.0，纯属路由行为而非检索退化）。
    router 本身的收益（准确率/成本/延迟）由 eval_router.py 单独度量，
    不混进这个 A/B。用 --use-router 可切换回真实路由做端到端对照。
    """

    def route(self, question: str) -> dict:
        return {
            "route": "retrieve",
            "reason": "forced (A/B 是检索路径评测)",
            "classifier": "forced",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="纠错闭环 A/B 评测")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=6, help="并发线程数（GPU 前向已加锁串行化）"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=540.0,
        help="时间预算（秒）。超预算当前块跑完即停，下次同 --tag 续跑",
    )
    parser.add_argument(
        "--tag", default="final", help="进度文件名后缀（固定用同一 tag 才能续跑）"
    )
    parser.add_argument(
        "--use-router",
        action="store_true",
        help="用真实查询路由（默认强制走检索路径，见 _ForceRetrieveRouter 说明）",
    )
    args = parser.parse_args()

    data = load_local()
    rel_map = defaultdict(set)
    for r in data["qrels"]:
        rel_map[r["qid"]].add(r["pid"])

    retriever = Retriever()
    router = None if args.use_router else _ForceRetrieveRouter()
    pipe = CorrectiveRAG(
        retriever=retriever, judge_threshold=JUDGE_THRESHOLD, router=router
    )
    targets = [q for q in data["queries"] if q["qid"] in rel_map]
    if args.limit:
        targets = targets[: args.limit]

    k = config.TOP_K_FINAL
    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = config.TRACE_DIR / f"ab_progress_{args.tag}.jsonl"

    done_qids = set()
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_qids.add(json.loads(line)["qid"])
    pending = [q for q in targets if q["qid"] not in done_qids]

    print(
        f"目标 {len(targets)} 条，已完成 {len(done_qids)} 条，本轮待跑 {len(pending)} 条，"
        f"并发 {args.workers}，预算 {args.budget:.0f}s",
        flush=True,
    )

    def _work(q: dict):
        """单条：朴素 top4 命中 + 闭环 pipeline，返回 (row, err)。"""
        relevant = rel_map[q["qid"]]
        naive_top = {r["id"] for r in retriever.retrieve(q["text"], top_k=k)}
        r = None
        last_err = None
        for attempt in range(MAX_RETRY):
            try:
                r = pipe.run(q["text"])
                break
            except Exception as e:  # noqa: BLE001 —— 网络抖动/限流，重试
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        if r is None:
            return None, last_err
        final_top = {c["id"] for c in r["ranked"]}
        return (
            {
                "qid": q["qid"],
                "query": q["text"],
                "status": r["status"],
                "iteration": r["iteration"],
                "naive_hit": 1.0 if naive_top & relevant else 0.0,
                "closed_hit": 1.0 if final_top & relevant else 0.0,
                "naive_recall": len(naive_top & relevant) / len(relevant),
                "closed_recall": len(final_top & relevant) / len(relevant),
                "judge_levels": [
                    t.get("level") for t in r["trace"] if t["step"] == "judge"
                ],
            },
            None,
        )

    t0 = time.time()
    processed = 0
    failed = 0
    progress_f = open(progress_path, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for start in range(0, len(pending), args.workers):
            if time.time() - t0 > args.budget:
                print(f"\n预算 {args.budget:.0f}s 用完，剩余条目下次同 --tag 续跑", flush=True)
                break
            chunk = pending[start : start + args.workers]
            for row, err in ex.map(_work, chunk):
                if row is None:
                    failed += 1
                    print(f"  重试 {MAX_RETRY} 次仍失败，跳过（下轮续跑会自动重试）：{err!r}",
                          flush=True)
                    continue
                progress_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                processed += 1
            progress_f.flush()
            print(
                f"  本轮已跑 {processed}/{len(pending)}，失败 {failed}，"
                f"用时 {time.time() - t0:.0f}s",
                flush=True,
            )
    progress_f.close()

    # 从进度文件全量重算：断点续跑跳过的条目也在文件里，聚合天然正确
    rows = []
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    n = len(rows)
    if n == 0:
        print("进度文件为空，不汇总")
        return
    answered_rows = [r for r in rows if r["status"] == "answered"]
    answered = len(answered_rows)

    from corrective_rag.judge import LLMJudge

    judge_desc = (
        f"llm({config.LLM_MODEL})"
        if isinstance(pipe.judge, LLMJudge)
        else f"rule-based(threshold={JUDGE_THRESHOLD})"
    )
    report = {
        "n_queries": n,
        "k": k,
        "judge": judge_desc,
        "workers": args.workers,
        "router": "real" if args.use_router else "forced-retrieve",
        "naive": {
            "recall@4": round(sum(r["naive_recall"] for r in rows) / n, 4),
            "hit@4": round(sum(r["naive_hit"] for r in rows) / n, 4),
        },
        "closed_loop": {
            "recall@4(answered only)": round(
                sum(r["closed_recall"] for r in answered_rows) / max(1, answered), 4
            ),
            "hit@4(answered only)": round(
                sum(r["closed_hit"] for r in answered_rows) / max(1, answered), 4
            ),
            "abstain_rate": round((n - answered) / n, 4),
            "avg_iterations": round(sum(r["iteration"] for r in rows) / n, 2),
        },
        "cost": {
            **tracker.snapshot(),
            # 注意：tracker 是**本进程**累计，而 n 是全量进度条数。分块跑时
            # 本块只跑了 processed 条，per-query 必须除以 processed 而非 n
            # （2026-10-13 踩到：最后一块 172 条除以 1000 得 ¥0.0006 假值）。
            "processed_this_run": processed,
            "cost_this_run_cny": round(tracker.cost_cny(config.LLM_MODEL), 4),
            "cny_per_query_this_run": round(
                tracker.cost_cny(config.LLM_MODEL) / max(1, processed), 5
            ),
        },
        "run_elapsed_sec": round(time.time() - t0, 1),
        "progress_file": str(progress_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    trace_path = (
        config.TRACE_DIR / f"corrective_ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"trace 已写入: {trace_path}")

    if processed:
        print(
            f"\n本轮新跑 {processed} 条，累计进度 {n}/{len(targets)}。"
            + ("" if n >= len(targets) else "仍有剩余，重跑同 --tag 继续。")
        )


if __name__ == "__main__":
    main()
