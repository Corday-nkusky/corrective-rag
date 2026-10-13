"""多轮对话记忆（Phase 5）：滑动窗口 + 超窗摘要。

为什么需要两级而不是全量塞历史：
- 全量塞：token 随轮数线性涨，第 10 轮时 prompt 里 90% 是旧对话，
  既贵又把模型的注意力稀释掉（lost in the middle）；
- 只要窗口：窗口外的指代丢失——"那个方案"指什么，模型只能猜；
- 滑动窗口 + 摘要：近 N 轮原文保真（细节、数字、实体名），
  超窗部分压成结构化摘要（关键实体 + 事实锚点），成本有界。

摘要为什么用 LLM 而不是截断：截断丢的是"用户后来纠正过的信息"
（"不是 A 是 B"），而纠正恰恰是最不能丢的。摘要 prompt 里明确
要求保留"纠正与变更"类事实。

checkpointer 说明：LangGraph 的 checkpointer 存的是图状态（可恢复
执行），不是对话历史；对话历史是应用层概念，由本模块管理。
两者互补，不重复。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402

SUMMARY_PROMPT = """把下面的旧对话压缩成结构化摘要。要求：
1. 保留所有关键实体名称、数字、日期、专有名词（原样，不要改写成"某公司"）
2. 显式保留"纠正与变更"类事实（用户说过"不是A是B"这种，最重要）
3. 保留用户的偏好与约束条件
4. 不超过 {max_chars} 字
5. 只输出摘要本身

旧对话：
{conversation}"""


class ConversationMemory:
    """会话记忆：近 window_size 轮原文 + 超窗部分的 LLM 摘要。

    - add_turn：每轮问答后调用（user question + assistant answer）
    - get_history：返回渲染好的历史文本，供 condense 节点使用；
      超窗部分只摘要一次并缓存，后续轮次直接复用（不重复花钱）
    """

    def __init__(self, window_size: int = 3, max_summary_chars: int = 300) -> None:
        self.window_size = window_size
        self.max_summary_chars = max_summary_chars
        self._turns: List[Dict[str, str]] = []  # [{"q": ..., "a": ...}]
        self._summary: str = ""  # 超窗部分的摘要（缓存）
        self._summarized_upto: int = 0  # 已摘要到第几轮（避免重复摘要）

    def add_turn(self, question: str, answer: str) -> None:
        self._turns.append({"q": question, "a": answer})

    @property
    def n_turns(self) -> int:
        return len(self._turns)

    def _render_turns(self, turns: List[Dict[str, str]]) -> str:
        return "\n".join(f"用户：{t['q']}\n助手：{t['a']}" for t in turns)

    def _summarize(self, conversation: str) -> str:
        """对话文本 → LLM 摘要。失败（无 key/超时）时降级为截断拼接，
        并在返回文本里标注原因——诚实降级，不假装摘要是有效的。"""
        if not config.LLM_API_KEY:
            return f"（无 LLM key，以下为截断）\n{conversation[: self.max_summary_chars]}"
        try:
            from openai import OpenAI

            from corrective_rag.cost import record

            client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": SUMMARY_PROMPT.format(
                            conversation=conversation,
                            max_chars=self.max_summary_chars,
                        ),
                    }
                ],
                temperature=0,
                max_tokens=400,
            )
            record(resp.usage, config.LLM_MODEL)
            summary = (resp.choices[0].message.content or "").strip()
            return summary or f"（摘要为空，以下为截断）\n{conversation[: self.max_summary_chars]}"
        except Exception as e:  # 摘要失败不能拖垮对话，降级
            return f"（摘要失败：{e}，以下为截断）\n{conversation[: self.max_summary_chars]}"

    def get_history(self) -> str:
        """渲染历史：超窗摘要（如有）+ 近 window_size 轮原文。空历史返回 ""。"""
        if not self._turns:
            return ""
        n_old = max(0, len(self._turns) - self.window_size)
        if n_old > self._summarized_upto:
            # 增量摘要：已有摘要 + 新增的超窗轮次一起压。
            # （全量重压在窗口 3 下 token 成本可忽略，简单可靠优先。）
            prior = f"（已有摘要）{self._summary}\n" if self._summary else ""
            new_old = self._turns[self._summarized_upto : n_old]
            self._summary = self._summarize(prior + self._render_turns(new_old))
            self._summarized_upto = n_old
        parts = []
        if self._summary:
            parts.append(f"【更早的对话摘要】\n{self._summary}")
        recent = self._turns[n_old:]
        if recent:
            parts.append(f"【最近的对话】\n{self._render_turns(recent)}")
        return "\n\n".join(parts)

    def reset(self) -> None:
        self._turns = []
        self._summary = ""
        self._summarized_upto = 0
