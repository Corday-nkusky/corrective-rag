"""路由评测（Phase 6）：router 分类准确率 + 直答路线收益。

两部分（口径不同，分开报，不混算）：

Part A · 自建标注集（data/eval/router_eval.jsonl，60 条：30 direct + 30 retrieve）
  - direct 类标签为自建（DuReader 是纯检索语料，没有"不需检索"的 query，
    这个类别只能手工造——已如实标注 source=hand，小样本，只证明链路正确）
  - retrieve 类标签 = 需要语料的具体事实/领域问题（与 DuReader 同分布）
  - 跑完整 pipeline，比对 router 决策与标签：准确率 / 混淆矩阵 /
    按路由分组的延迟与成本（直答应省掉 rerank CPU + judge 一次调用）

Part B · DuReader dev 前 N 条（默认 100）
  - 标签 = retrieve（构造性：真实搜索日志来的 query，全部需要检索）
  - 只分类不生成（便宜），看误判 direct 的比例——这是"路由不破坏主链路"
    的证据。边界 case 如实报（如"国家法定节假日共多少天"这种稳定常识型
    搜索词，判 direct 不算错得离谱，单独列出人工看）

用法：
    PYTHONPATH=src python scripts/eval_router.py [--limit 100] [--skip-part-b]

trace 落 data/traces/router_eval_*.json；逐条进度落同目录 .jsonl（断线重跑
不重复花钱，汇总从全量文件重算）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.cost import tracker  # noqa: E402
from corrective_rag.pipeline import CorrectiveRAG  # noqa: E402

EVAL_SET = config.EVAL_DIR / "router_eval.jsonl"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="路由评测")
    parser.add_argument("--limit", type=int, default=100, help="Part B 取 DuReader dev 前 N 条")
    parser.add_argument("--skip-part-b", action="store_true", help="只跑 Part A")
    parser.add_argument(
        "--budget",
        type=float,
        default=540.0,
        help="时间预算（秒）。超预算即停，剩余条目下次运行自动续跑（进度逐条落盘）",
    )
    args = parser.parse_args()

    labeled = load_jsonl(EVAL_SET)
    du_queries = []
    if not args.skip_part_b:
        du_all = load_jsonl(config.EVAL_DIR / "dureader_queries.jsonl")
        du_queries = du_all[: args.limit]
    if not labeled and not du_queries:
        print("没有评测数据，退出")
        return

    print("加载 pipeline（模型 + 100K 索引）...")
    pipe = CorrectiveRAG()
    router = pipe.router
    print(f"分类器：{type(router).__name__}")

    # 固定进度文件名：跨运行续跑靠它（完整跑完才改名归档成 .done）
    progress_path = config.TRACE_DIR / "router_eval.progress.jsonl"
    done = load_jsonl(progress_path) if progress_path.exists() else []
    done_keys = {(d["part"], d["qid"]) for d in done}

    # ---------------- Part A：完整 pipeline ----------------
    print(f"\n== Part A：自建标注集 {len(labeled)} 条（完整 pipeline）==")
    t_start = time.time()
    with progress_path.open("a", encoding="utf-8") as f:
        for item in labeled:
            key = ("A", item["qid"])
            if key in done_keys:
                continue
            if time.time() - t_start > args.budget:
                print(f"\n预算 {args.budget:.0f}s 用完，Part A 剩余条目下次运行自动续跑")
                return
            cost0 = tracker.cost_cny(config.LLM_MODEL)
            calls0 = tracker.calls
            t0 = time.time()
            r = pipe.run(item["query"])
            elapsed = time.time() - t0
            rec = {
                "part": "A",
                "qid": item["qid"],
                "query": item["query"],
                "label": item["label"],
                "route": r.get("route"),
                "status": r.get("status"),
                "iteration": r.get("iteration"),
                "elapsed": round(elapsed, 2),
                "cost": round(tracker.cost_cny(config.LLM_MODEL) - cost0, 5),
                "calls": tracker.calls - calls0,
                "answer_head": (r.get("answer") or "")[:80],
            }
            # router trace 里的理由
            for t in r.get("trace", []):
                if t["step"] == "router":
                    rec["router_reason"] = t.get("reason")
                    rec["classifier"] = t.get("classifier")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            mark = "✓" if rec["route"] == rec["label"] else "✗"
            print(f"  {mark} [{rec['route']:8s}] {item['query'][:22]}"
                  f"｜{rec['status']}｜{rec['elapsed']}s｜¥{rec['cost']}")
    print(f"Part A 耗时 {time.time() - t_start:.0f}s")

    # ---------------- Part B：只分类 ----------------
    if du_queries:
        print(f"\n== Part B：DuReader dev 前 {len(du_queries)} 条（只分类，不生成）==")
        with progress_path.open("a", encoding="utf-8") as f:
            for item in du_queries:
                key = ("B", item["qid"])
                if key in done_keys:
                    continue
                if time.time() - t_start > args.budget:
                    print(f"\n预算 {args.budget:.0f}s 用完，Part B 剩余条目下次运行自动续跑")
                    return
                cost0 = tracker.cost_cny(config.LLM_MODEL)
                t0 = time.time()
                d = router.route(item["text"])
                rec = {
                    "part": "B",
                    "qid": item["qid"],
                    "query": item["text"],
                    "label": "retrieve",
                    "route": d.get("route"),
                    "router_reason": d.get("reason"),
                    "classifier": d.get("classifier"),
                    "elapsed": round(time.time() - t0, 2),
                    "cost": round(tracker.cost_cny(config.LLM_MODEL) - cost0, 5),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if rec["route"] != "retrieve":
                    print(f"  ✗ 误判 direct：{item['text']}｜理由：{rec['router_reason']}")

    # ---------------- 汇总（从全量进度文件重算） ----------------
    records = load_jsonl(progress_path)
    a = [r for r in records if r["part"] == "A"]
    b = [r for r in records if r["part"] == "B"]

    def confusion(items):
        # 正类 = retrieve
        tp = sum(1 for r in items if r["label"] == "retrieve" and r["route"] == "retrieve")
        fn = sum(1 for r in items if r["label"] == "retrieve" and r["route"] != "retrieve")
        tn = sum(1 for r in items if r["label"] == "direct" and r["route"] == "direct")
        fp = sum(1 for r in items if r["label"] == "direct" and r["route"] != "direct")
        return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}

    def by_route(items, route):
        sel = [r for r in items if r["route"] == route]
        if not sel:
            return {"n": 0}
        return {
            "n": len(sel),
            "avg_elapsed": round(sum(r["elapsed"] for r in sel) / len(sel), 2),
            "avg_cost": round(sum(r["cost"] for r in sel) / len(sel), 5),
            "avg_calls": round(sum(r["calls"] for r in sel) / len(sel), 2),
            "answered": sum(1 for r in sel if r["status"] == "answered"),
            "abstained": sum(1 for r in sel if r["status"] == "abstained"),
        }

    summary = {
        "meta": {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "classifier": type(router).__name__,
            "model": config.LLM_MODEL,
            "part_a_n": len(a),
            "part_b_n": len(b),
            "total_cost_cny": round(tracker.cost_cny(config.LLM_MODEL), 4),
        },
        "part_a": {
            "confusion": confusion(a),
            "accuracy": round(sum(1 for r in a if r["route"] == r["label"]) / len(a), 4) if a else None,
            "direct_recall": round(
                sum(1 for r in a if r["label"] == "direct" and r["route"] == "direct")
                / max(1, sum(1 for r in a if r["label"] == "direct")), 4),
            "retrieve_recall": round(
                sum(1 for r in a if r["label"] == "retrieve" and r["route"] == "retrieve")
                / max(1, sum(1 for r in a if r["label"] == "retrieve")), 4),
            "by_route": {
                "direct": by_route(a, "direct"),
                "retrieve": by_route(a, "retrieve"),
            },
            "misclassified": [
                {"query": r["query"], "label": r["label"], "route": r["route"],
                 "reason": r.get("router_reason")}
                for r in a if r["route"] != r["label"]
            ],
        },
        "part_b": {
            "n": len(b),
            "false_direct": sum(1 for r in b if r["route"] == "direct"),
            "false_direct_rate": round(
                sum(1 for r in b if r["route"] == "direct") / len(b), 4) if b else None,
            "false_direct_items": [
                {"query": r["query"], "reason": r.get("router_reason")}
                for r in b if r["route"] == "direct"
            ],
            "avg_cost": round(sum(r["cost"] for r in b) / len(b), 5) if b else None,
        },
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = config.TRACE_DIR / f"router_eval_{ts}.json"
    trace_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress_path.rename(trace_path.with_suffix(".jsonl.done"))

    print("\n===== 汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\ntrace: {trace_path}")


if __name__ == "__main__":
    main()
