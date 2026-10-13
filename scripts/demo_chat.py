"""多轮对话 demo（Phase 5）：交互式 CLI，基于真实 DuReader 100K 索引。

用法：
    PYTHONPATH=src python scripts/demo_chat.py
    PYTHONPATH=src python scripts/demo_chat.py --scripted   # 跑内置 5 轮指代追问脚本（无交互，适合演示/验证）

设计要点：
- ConversationMemory 管历史（滑动窗口 3 轮 + 超窗 LLM 摘要）
- pipeline.run(question, history) —— history 空则 condense 透传不花钱
- 每轮结束打印：状态（answered/abstained）、迭代数、本轮检索命中情况
- /reset 清空会话，/quit 退出
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.cost import tracker  # noqa: E402
from corrective_rag.memory import ConversationMemory  # noqa: E402
from corrective_rag.pipeline import CorrectiveRAG  # noqa: E402

# 内置指代追问脚本（演示用）：主题均为 DuReader 语料覆盖的生活类问题，
# T2/T3 是纯指代（condense 必须结合历史才能搜对），T4 是实体切换
# （考察 condense 不被历史带偏），T5 是历史+指代组合
SCRIPTED_TURNS = [
    "感冒了怎么办？",
    "那吃什么药好？",
    "它多久能好？",
    "孕妇感冒了呢？",
    "刚才说的药，孕妇能吃吗？",
]


def print_turn(i: int, question: str, r: dict, elapsed: float) -> None:
    status = r["status"]
    mark = "✓" if status == "answered" else "✗"
    print(f"\n[{i}] {mark} 问：{question}")
    print(f"    状态：{status}｜迭代 {r['iteration']} 次｜耗时 {elapsed:.1f}s")
    steps = [t["step"] for t in r["trace"]]
    print(f"    路径：{' → '.join(steps)}")
    router = [t for t in r["trace"] if t["step"] == "router"]
    if router:
        tag = "直答（跳过检索）" if router[0].get("route") == "direct" else "检索（纠错循环）"
        print(f"    路由：{tag}｜{router[0].get('reason')}")
    condense = [t for t in r["trace"] if t["step"] == "condense"]
    if condense and condense[0].get("applied"):
        print(f"    condense：{condense[0].get('new_query')}")
    answer = r["answer"].strip()
    print(f"    答：{answer[:200]}{'...' if len(answer) > 200 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮对话 demo")
    parser.add_argument("--scripted", action="store_true", help="跑内置脚本，无交互")
    parser.add_argument("--window", type=int, default=3, help="滑动窗口轮数")
    args = parser.parse_args()

    print("加载索引与模型...")
    pipe = CorrectiveRAG()
    memory = ConversationMemory(window_size=args.window)
    print(f"就绪（窗口 {args.window} 轮 + 超窗摘要）。累计已花费 ¥{tracker.cost_cny(config.LLM_MODEL):.3f}")
    print("命令：/reset 清空会话｜/quit 退出\n")

    if args.scripted:
        for i, q in enumerate(SCRIPTED_TURNS, start=1):
            history = memory.get_history()
            t0 = time.time()
            r = pipe.run(q, history=history)
            print_turn(i, q, r, time.time() - t0)
            memory.add_turn(q, r["answer"])
        print(f"\n脚本跑完。累计 {memory.n_turns} 轮，"
              f"总花费 ¥{tracker.cost_cny(config.LLM_MODEL):.3f}")
        return

    turn = 0
    while True:
        try:
            q = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/reset":
            memory.reset()
            print("会话已清空。")
            continue
        turn += 1
        history = memory.get_history()
        t0 = time.time()
        r = pipe.run(q, history=history)
        print_turn(turn, q, r, time.time() - t0)
        memory.add_turn(q, r["answer"])


if __name__ == "__main__":
    main()
