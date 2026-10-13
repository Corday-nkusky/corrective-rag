"""网页版对话 demo（Phase 6）：Gradio 界面 + 纠错过程可视化。

用法：
    PYTHONPATH=src python scripts/demo_web.py                # 启动预热模型，开 7860 端口
    PYTHONPATH=src python scripts/demo_web.py --lazy         # 首次请求时才加载（启动快）
    PYTHONPATH=src python scripts/demo_web.py --port 7861    # 换端口

与 CLI 版（demo_chat.py）的关系：同一套 CorrectiveRAG + ConversationMemory，
本脚本只加一层 Web 界面，并把 trace / 检索片段 / 成本渲染成可视化面板。
CLI 版适合脚本化验证，网页版适合给人看（面试演示、项目展示）。

设计要点：
1. 模型与索引默认在启动时预热（演示体验优先：第一个问题不用干等 30 秒），
   --lazy 则推迟到首次请求；
2. pipeline 单例全局共享（只读，加载一次约 20-40s）；会话记忆放
   gradio State，按浏览器会话隔离——开两个标签页互不串历史；
3. 只绑 127.0.0.1、不开 share：服务端持有 MiMo API key，
   公网分享等于把 key 送给所有访问者；
4. 每轮右侧面板展示：状态（answered/abstained）、状态机路径、
   condense 指代消解、judge 判定理由、rewrite 重试、Top4 检索片段、
   本轮与累计成本——"纠错"不是口号，每一步都看得见。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.cost import tracker  # noqa: E402
from corrective_rag.memory import ConversationMemory  # noqa: E402
from corrective_rag.pipeline import CorrectiveRAG  # noqa: E402

# ---------------------------------------------------------------------------
# pipeline 单例（懒加载 + 双检锁；gradio 多线程回调下也只加载一次）
# ---------------------------------------------------------------------------
_PIPE: CorrectiveRAG | None = None
_PIPE_LOCK = threading.Lock()
_LOAD_SECONDS: float = 0.0


def get_pipeline() -> CorrectiveRAG:
    global _PIPE, _LOAD_SECONDS
    if _PIPE is None:
        with _PIPE_LOCK:
            if _PIPE is None:
                t0 = time.time()
                _PIPE = CorrectiveRAG()
                _LOAD_SECONDS = time.time() - t0
    return _PIPE


# ---------------------------------------------------------------------------
# 面板渲染
# ---------------------------------------------------------------------------
_IDLE_DETAIL = """### 本轮过程

发送问题后，这里会显示纠错全链路：

- **状态**：answered / abstained（诚实拒答）
- **路径**：状态机实际走过的节点（condense → retrieve → rerank → judge → …）
- **condense**：多轮追问的指代消解（"那它呢" → 独立问题）
- **judge**：资料充分性判定与理由（不充分会 rewrite 重试或拒答）
- **Top4 片段**：实际进入生成 prompt 的检索结果（rerank 分）
- **成本**：本轮与累计 API 花费（按 MiMo 官方价目表实时核算）

试试连续追问（指代消解是重点）：
1. 感冒了怎么办？
2. 那吃什么药好？
3. 孕妇感冒了呢？
"""


def _detail_md(
    r: dict,
    elapsed: float,
    d_calls: int,
    d_cost: float,
    total_cost: float,
    memory: ConversationMemory,
) -> str:
    steps = [t["step"] for t in r["trace"]]
    answered = r["status"] == "answered"
    lines = [
        f"### 第 {memory.n_turns} 轮 · {'✓ 已回答' if answered else '✗ 诚实拒答'}",
        "",
        f"- **路径**：{' → '.join(steps)}",
        f"- **迭代**：{r['iteration']} 次（上限 {config.MAX_ITERATIONS}）"
        f"｜**耗时** {elapsed:.1f}s",
        f"- **LLM 调用**：{d_calls} 次｜**本轮花费** ¥{d_cost:.4f}"
        f"｜**累计** ¥{total_cost:.4f}",
    ]
    for t in r["trace"]:
        step = t["step"]
        if step == "condense" and t.get("applied"):
            lines.append(f"- **condense**：`{r['question']}` → `{t.get('new_query')}`")
        elif step == "judge":
            mark = "✓ 资料充分" if t.get("sufficient") else "✗ 资料不足"
            lines.append(f"- **judge**：{mark}｜{t.get('reason')}")
        elif step == "rewrite" and t.get("applied"):
            lines.append(
                f"- **rewrite**（第 {t.get('iteration')} 次重试）：→ `{t.get('new_query')}`"
            )
    ranked = r.get("ranked") or []
    if ranked:
        lines += ["", "**进入生成的 Top 片段**（格式：rerank 分｜片段）："]
        for i, d in enumerate(ranked[:4], 1):
            snippet = d.get("text", "").replace("\n", " ")[:80]
            lines.append(
                f"{i}. `{d.get('rerank_score')}`｜{d.get('source', '?')}｜{snippet}…"
            )
    if memory.n_turns > memory.window_size:
        lines += ["", f"（历史已超窗口 {memory.window_size} 轮，更早部分已压成摘要）"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 聊天回调
# ---------------------------------------------------------------------------
def respond(message: str, chat: list, memory: ConversationMemory):
    """gradio 回调：输入框 + 历史 + 会话记忆 → 新历史、过程面板、清空输入框。"""
    message = (message or "").strip()
    if not message:
        return chat, _IDLE_DETAIL, ""

    pipe = get_pipeline()  # 首次调用在此加载模型与索引（--lazy 模式下）
    history = memory.get_history()
    calls0 = tracker.calls
    cost0 = tracker.cost_cny(config.LLM_MODEL)
    t0 = time.time()
    try:
        r = pipe.run(message, history=history)
    except Exception as e:  # demo 不该被单次异常打爆，错误也展示出来
        chat = (chat or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"（系统错误：{type(e).__name__}: {e}）"},
        ]
        return chat, f"### 本轮出错\n\n`{type(e).__name__}: {e}`", ""
    elapsed = time.time() - t0

    d_calls = tracker.calls - calls0
    d_cost = tracker.cost_cny(config.LLM_MODEL) - cost0
    memory.add_turn(message, r["answer"])
    chat = (chat or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": r["answer"]},
    ]
    return chat, _detail_md(r, elapsed, d_calls, d_cost, tracker.cost_cny(config.LLM_MODEL), memory), ""


def _reset(window: int):
    return [], _IDLE_DETAIL, ConversationMemory(window_size=window)


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
_HEADER_MD = """## 纠错式 RAG 对话 Demo

**不是"检索到就生成"**：检索 → 重排 → **LLM 裁判判定资料够不够** → 不够就
改查询重试（最多 {max_iter} 次）→ 还不够就**诚实拒答**。多轮对话里，
"那吃什么药好？"这类指代会先被 condense 成独立问题再检索。

- 知识库：DuReader-retrieval 真实语料 **100,001 段**
- 检索：bge-small-zh 向量召回 top20 → bge-reranker 精选 top4
- LLM：MiMo-v2.6-flash（裁判 / 改写 / 生成同源，成本实时显示）
""".format(max_iter=config.MAX_ITERATIONS)


def build_demo(window: int):
    import gradio as gr

    with gr.Blocks(title="纠错式 RAG Demo") as demo:
        gr.Markdown(_HEADER_MD)
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="问我任何一个生活类问题；多轮追问可以试指代消解",
                )
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="输入问题，回车发送",
                        scale=8,
                        container=False,
                        autofocus=True,
                    )
                    send = gr.Button("发送", variant="primary", scale=1)
                clear = gr.Button("清空会话", size="sm")
            with gr.Column(scale=1):
                detail = gr.Markdown(_IDLE_DETAIL)
        memory = gr.State(ConversationMemory(window_size=window))

        send.click(respond, [textbox, chatbot, memory], [chatbot, detail, textbox])
        textbox.submit(respond, [textbox, chatbot, memory], [chatbot, detail, textbox])
        clear.click(
            lambda: _reset(window), None, [chatbot, detail, memory]
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="纠错式 RAG 网页 demo")
    parser.add_argument("--port", type=int, default=7860, help="端口（默认 7860）")
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="首次请求时才加载模型与索引（默认启动时预热，演示体验优先）",
    )
    parser.add_argument("--window", type=int, default=3, help="对话记忆滑动窗口轮数")
    args = parser.parse_args()

    if not args.lazy:
        t0 = time.time()
        get_pipeline()
        print(
            f"模型与索引已预热（{time.time() - t0:.1f}s）。"
            f"进程累计已花费 ¥{tracker.cost_cny(config.LLM_MODEL):.4f}"
        )
    else:
        print("懒加载模式：首次提问时才会加载模型与索引（约 20-40 秒）。")

    demo = build_demo(args.window)
    # 只绑回环地址：服务端持有 MiMo key，不对公网开放
    demo.launch(server_name="127.0.0.1", server_port=args.port, inbrowser=True)


if __name__ == "__main__":
    main()
