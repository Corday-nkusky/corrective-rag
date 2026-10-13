"""答案生成：检索结果 + 问题 → MiMo → 带引用的答案（Phase 1 组件）。

设计要点：
- stub 模式：没有 LLM_API_KEY 也能验证"检索→拼 prompt"链路
  （打印将要发送的 messages）， Phase 1 冒烟和离线调试都靠它。
- prompt 三条硬规矩：只许用给定资料、每个论点标 [n] 引用、
  资料不足必须明说"根据现有资料无法回答"——不许自由发挥。
  第三条是 Phase 2 拒答能力的地基：生成端先立规矩，judge 端才能测出来。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402

SYSTEM_PROMPT = """你是一个严格基于给定资料回答问题的助手。规则：
1. 只能使用下面列出的资料内容回答，禁止使用你自己的知识补充。
2. 回答中每个事实性论点必须在句末标注引用编号，如[1]、[2]。
3. 如果资料不足以回答问题，明确回答"根据现有资料无法回答"，不要猜测。
4. 回答用简体中文，直接给出结论，不要复述问题。"""


def build_messages(
    question: str,
    contexts: List[Dict],
    resolved_query: str = "",
) -> List[Dict[str, str]]:
    """把检索到的 chunk 拼成编号资料块，组装成 chat messages。

    resolved_query：condense 消解指代后的查询（多轮场景）。传入时
    prompt 里显式说明"用户问的 X 理解为 Y"——否则生成器看到"它多久
    能好"会装不知道"它"是什么，而系统自己刚消解过（Phase 5 demo
    实测闹过这个笑话：condense 出"风热感冒"，答案说"未指明它指哪种
    疾病"）。单轮时为空，行为不变。
    """
    if contexts:
        blocks = []
        for i, c in enumerate(contexts, start=1):
            src = c.get("source", "unknown")
            blocks.append(f"[{i}] (来源: {src})\n{c['text']}")
        user_content = f"资料：\n{chr(10).join(blocks)}\n\n问题：{question}"
    else:
        user_content = f"问题：{question}\n（未检索到任何资料）"
    if resolved_query and resolved_query != question:
        user_content += (
            f"\n\n（注：用户的问题已结合对话历史消解指代，"
            f"按\"{resolved_query}\"理解作答）"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate(
    question: str,
    contexts: List[Dict],
    stub: Optional[bool] = None,
    resolved_query: str = "",
) -> str:
    """生成答案。stub=True 时不调 API，返回将要发送的内容（供冒烟/调试）。"""
    messages = build_messages(question, contexts, resolved_query=resolved_query)

    if stub is None:
        stub = not config.LLM_API_KEY
    if stub:
        print("[generate] stub 模式：未调用 LLM API，以下为将发送的 messages")
        print("=" * 60)
        for m in messages:
            print(f"--- {m['role']} ---")
            print(m["content"])
        print("=" * 60)
        return "[stub] 未生成真实答案（无 API key 或显式 stub）"

    from openai import OpenAI

    from corrective_rag.cost import record

    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )
    record(resp.usage, config.LLM_MODEL)
    return resp.choices[0].message.content or ""
