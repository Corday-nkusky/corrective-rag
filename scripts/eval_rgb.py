"""RGB 四场景评测：负例拒绝 / 噪声鲁棒 / 信息整合 / 反事实鲁棒（Phase 4）。

为什么需要它：DuReader 每个 query 都有标注相关段——基准里根本没有"应该
拒答"的样本，闭环的 judge/abstain 分支在它上面永远不触发（Phase 3 A/B
实测 abstain=0）。RGB 是公开基准里少见的带"应拒答/含噪/伪事实"构造的，
正好补上这块。

场景与输入构造（noise_rate = 噪声文档占输入比例，对齐官方语义）：
- rejection:      全 negative（noise_rate=1）→ 期望系统 abstain
- noise:          positives + negatives×(0.6/0.4) → 期望答对
- integration:    positives + negatives×(0.6/0.4) → 期望覆盖两个要点
- counterfactual: fake 文档（positive_wrong，说假答案的）+ negatives×1.5
                  → 期望答真答案而非被 fake 带偏

指标（对齐官方 evalue.py / reject_evalue.py / fact_evalue.py 口径）：
- rejection 场景: abstain_rate（拒答率）；幻觉率 = 未拒答且答案不含要点
- noise 场景: accuracy = 答案包含全部要点
- integration 场景: accuracy（覆盖两个要点）+ partial_rate（至少覆盖一个）
- counterfactual 场景: 被骗率（答案含 fake_answer）+ 纠正率（答案含真答案）
  （官方 ED*/CR 需要模型显式指出错误，我们的生成 prompt 没要求显式纠错，
  用"答对真答案"作 CR 近似、'答成 fake' 作被骗——口径差异写进报告）

判定函数：
- is_rejection(text)：命中拒绝关键词（无法回答/信息不足/资料不足/无法确定……）
- covered(text, keys)：每个 key 去空白后为 text 去空白后的子串

用法：
    PYTHONPATH=src python scripts/eval_rgb.py --scenario rejection --limit 50
    PYTHONPATH=src python scripts/eval_rgb.py --scenario noise --limit 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.cost import tracker  # noqa: E402
from corrective_rag.data_loader import load_rgb_local  # noqa: E402
from corrective_rag.pipeline import CorrectiveRAG  # noqa: E402
from corrective_rag.retrieve import FixedRetriever  # noqa: E402

REJECTION_MARKERS = [
    "无法回答",
    "信息不足",
    "资料不足",
    "无法确定",
    "没有提到",
    "未提及",
    " insufficient",
    "无法判断",
    "不能确定",
    "缺乏足够",
    "资料中没",
    "资料里没",
]


def is_rejection(text: str) -> bool:
    return any(m in text for m in REJECTION_MARKERS)


def covered(text: str, keys: List[str]) -> bool:
    """所有要点都是答案的子串（去空白后比较）。"""
    if not keys:
        return False
    t = "".join(text.split())
    return all("".join(str(k).split()) in t for k in keys if str(k).strip())


def partial_covered(text: str, keys: List[str]) -> bool:
    t = "".join(text.split())
    return any("".join(str(k).split()) in t for k in keys if str(k).strip())


def build_input_docs(sample: Dict, scenario: str, noise_rate: float = 0.6) -> List[Dict]:
    """按场景构造输入文档。noise_rate = 噪声占输入比例（官方语义）。

    采样用固定 seed（按样本 id），保证可复现——面试问"你的噪声怎么加的"
    要能答"固定种子可复现"，不是随口说的。
    """
    docs = sample["docs"]
    pos = [d for d in docs if d["role"] == "positive"]
    neg = [d for d in docs if d["role"] == "negative"]
    fake = [d for d in docs if d["role"] == "fake"]

    rng = random.Random(zlib.crc32(sample["id"].encode("utf-8")))  # 稳定种子：跨进程可复现

    if scenario == "rejection":
        return neg  # noise_rate=1：全噪声
    if scenario == "noise":
        n_neg = int(round(len(pos) * noise_rate / (1 - noise_rate)))
        return pos + rng.sample(neg, min(n_neg, len(neg)))
    if scenario == "integration":
        n_neg = int(round(len(pos) * noise_rate / (1 - noise_rate)))
        return pos + rng.sample(neg, min(n_neg, len(neg)))
    if scenario == "counterfactual":
        # 官方 fact 场景：输入以伪事实文档为主（positive_wrong）
        n_neg = int(round(len(fake) * noise_rate / (1 - noise_rate)))
        return fake + rng.sample(neg, min(n_neg, len(neg)))
    raise ValueError(f"unknown scenario: {scenario}")


def run_scenario(scenario: str, limit: int | None, judge_threshold: float = 0.0) -> Dict:
    subset_map = {
        "rejection": "zh",
        "noise": "zh",
        "integration": "zh_int",
        "counterfactual": "zh_fact",
    }
    samples = load_rgb_local(subset_map[scenario])
    if limit:
        samples = samples[:limit]

    n = 0
    abstained = 0
    correct = 0
    partial = 0
    fooled = 0  # counterfactual：答案被 fake 带偏
    iters = 0
    rows = []
    t0 = time.time()

    for i, s in enumerate(samples):
        docs = build_input_docs(s, scenario)
        pipe = CorrectiveRAG(
            retriever=FixedRetriever(docs), judge_threshold=judge_threshold
        )
        r = pipe.run(s["query"])
        answer = r["answer"]
        keys = s["answer"]

        is_rej = is_rejection(answer)
        row = {
            "id": s["id"],
            "answer_head": answer[:60],
            "status": r["status"],
            "iterations": r["iteration"],
        }

        if scenario == "rejection":
            if r["status"] == "abstained":
                abstained += 1
            elif not is_rej and covered(answer, keys):
                correct += 1  # 全噪声下居然答对了（运气好，噪声里恰好有）
            if not is_rej and not covered(answer, keys):
                row["hallucinated"] = True
        elif scenario == "counterfactual":
            if covered(answer, keys):
                correct += 1
            fake_key = "".join(str(s["fake_answer"]).split())
            if fake_key and fake_key in "".join(answer.split()):
                fooled += 1
        else:
            if covered(answer, keys):
                correct += 1
            if partial_covered(answer, keys):
                partial += 1

        iters += r["iteration"]
        rows.append(row)
        n += 1
        if (i + 1) % 20 == 0:
            print(
                f"  {i + 1}/{len(samples)}，耗时 {time.time() - t0:.0f}s，"
                f"累计花费 ¥{tracker.cost_cny(config.LLM_MODEL):.3f}"
            )

    elapsed = time.time() - t0
    report: Dict = {
        "scenario": scenario,
        "subset": subset_map[scenario],
        "n": n,
        "judge": "llm" if config.LLM_API_KEY else "rule",
        "avg_iterations": round(iters / max(1, n), 2),
        "cost": {
            **tracker.snapshot(),
            "cny_per_query": round(tracker.cost_cny(config.LLM_MODEL) / max(1, n), 5),
        },
        "elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if scenario == "rejection":
        report["abstain_rate"] = round(abstained / max(1, n), 4)
        report["answered_correctly"] = round(correct / max(1, n), 4)
        report["hallucination_rate"] = round(
            sum(1 for r in rows if r.get("hallucinated")) / max(1, n), 4
        )
    elif scenario == "counterfactual":
        report["correct_rate_CR"] = round(correct / max(1, n), 4)
        report["fooled_rate"] = round(fooled / max(1, n), 4)
    elif scenario == "integration":
        report["accuracy_both_parts"] = round(correct / max(1, n), 4)
        report["partial_rate"] = round(partial / max(1, n), 4)
    else:
        report["accuracy"] = round(correct / max(1, n), 4)

    report["samples"] = rows
    print(json.dumps({k: v for k, v in report.items() if k != "samples"},
                     ensure_ascii=False, indent=2))

    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = (
        config.TRACE_DIR
        / f"rgb_{scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"trace 已写入: {trace_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RGB 四场景评测")
    parser.add_argument(
        "--scenario",
        choices=["rejection", "noise", "integration", "counterfactual"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_scenario(args.scenario, args.limit)


if __name__ == "__main__":
    main()
