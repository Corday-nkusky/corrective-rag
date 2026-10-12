"""Phase 3 闭环冒烟：状态机三用例 + 图导出（真实索引，非 mock）。

用例 1：DuReader 里标注可答的 query → 期望 answered
用例 2：语料里没有的 query → 期望走 rewrite 循环后 abstain（或 answered，
        重点验证 trace 完整、有终态、iteration 受控）
用例 3：导出 mermaid 图，确认 LangGraph 图结构正确

用法：
    PYTHONPATH=src python scripts/smoke_corrective.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag.data_loader import load_local  # noqa: E402
from corrective_rag.pipeline import CorrectiveRAG  # noqa: E402


def main() -> None:
    data = load_local()
    queries = {q["qid"]: q["text"] for q in data["queries"]}
    answerable = queries[data["queries"][0]["qid"]]

    pipe = CorrectiveRAG()

    # 用例 1：可答
    r1 = pipe.run(answerable)
    print(f"[用例1 可答] status={r1['status']} iter={r1['iteration']}")
    print(f"  answer: {r1['answer'][:80]}")
    print(f"  trace: {[t['step'] for t in r1['trace']]}")
    assert r1["status"] == "answered", "可答 query 未生成答案"

    # 用例 2：语料里没有
    r2 = pipe.run("南开大学食堂的糖醋排骨多少钱一份")
    print(f"\n[用例2 无关] status={r2['status']} iter={r2['iteration']}")
    print(f"  answer: {r2['answer'][:80]}")
    print(f"  trace: {[t['step'] for t in r2['trace']]}")
    assert r2["status"] in ("answered", "abstained"), "无终态"
    assert r2["iteration"] <= 3, "iteration 失控"
    steps = [t["step"] for t in r2["trace"]]
    if r2["status"] == "abstained":
        assert "abstain" in steps, "拒答路径缺 abstain 步骤"

    # 用例 3：图导出
    mermaid = pipe.draw()
    print(f"\n[用例3 图导出] {len(mermaid)} 字符")
    assert "retrieve" in mermaid and "abstain" in mermaid and "judge" in mermaid
    print(mermaid)

    print("\nSMOKE OK - Phase 3 闭环状态机通过")


if __name__ == "__main__":
    main()
