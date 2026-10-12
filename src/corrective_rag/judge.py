"""充分性判断（Phase 3 第二层质检，整个系统的质量门控）。

它和 reranker 回答的不是同一个问题：
- reranker：每篇文档与问题**相关不相关**（逐篇、局部）
- judge：检索到的文档**合起来够不够回答这个问题**（整体、全局）
  三篇各讲一半事实的文档，逐篇都"相关"，合起来答不了——
  这个区分是 CRAG 的 evaluator 之后 RAG 评测的核心设计。

口径：0/1/2（不够 / 边缘 / 够），不用 1-10。LLM 打 1-10 分校准极差，
分布堆在 7-8，阈值切上去噪声大；三档足够表达门控决策且稳定。

两种实现，同一接口（有 key 用 LLM，没 key 用规则跑通链路）：
- LLMJudge：MiMo 结构化输出，只判充分性（相关性已由 reranker 保证）
- RuleBasedJudge：基于 reranker 分数的规则基线。两个作用：
  ① 无 API key 时全链路可跑；② 作为 LLM judge 的对照——
  "我的 LLM judge 比规则基线多救回多少"本身就是面试素材。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402

JUDGE_PROMPT = """你是检索结果质量评估员。给你一个问题和若干检索到的资料，判断这些资料**合起来**是否足以回答问题。

注意：是整体判断，不是逐篇判断。资料各自相关但合起来覆盖不了问题，算"不够"。

问题：{question}

资料：
{contexts}

只输出 JSON，不要输出任何其他内容：
{{"level": 0或1或2, "reason": "一句话理由"}}
level 含义：0=不够（缺关键信息），1=边缘（能答一部分），2=够（能完整回答）"""


class RuleBasedJudge:
    """无 LLM 的规则门控：top1 rerank 分数低于阈值即判不充分。

    阈值从数据校准（scripts/calibrate_judge.py），不拍脑袋——
    用 DuReader 的人工标注把"能答的 query"和"答不了的 query"分开，
    看两类 query 的 top1 rerank 分数分布，取分界点。
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def judge(self, question: str, chunks: List[Dict]) -> Dict:
        if not chunks:
            return {"level": 0, "sufficient": False, "reason": "无检索结果"}
        top = chunks[0].get("rerank_score")
        if top is None:
            return {"level": 0, "sufficient": False, "reason": "无 rerank 分数（链路断）"}
        if top >= self.threshold:
            level = 2
        elif top >= self.threshold - 2.0:
            level = 1
        else:
            level = 0
        return {
            "level": level,
            "sufficient": level >= config.JUDGE_RELEVANCE_THRESHOLD,
            "reason": f"top1_rerank={top:.3f} vs threshold={self.threshold:.3f}",
        }


class LLMJudge:
    """MiMo 充分性判断。judge 与生成器异源是显水平细节
    （同源模型倾向于认可自己生成的上下文，自评偏差大）。"""

    def __init__(self) -> None:
        if not config.LLM_API_KEY:
            raise RuntimeError("LLMJudge 需要 LLM_API_KEY（或 DEEPSEEK_API_KEY）")

    def judge(self, question: str, chunks: List[Dict]) -> Dict:
        from openai import OpenAI

        from corrective_rag.cost import record

        blocks = []
        for i, c in enumerate(chunks, start=1):
            blocks.append(f"[{i}] {c['text']}")
        prompt = JUDGE_PROMPT.format(
            question=question, contexts="\n\n".join(blocks) or "（无）"
        )
        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        record(resp.usage, config.LLM_MODEL)
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            level = int(parsed.get("level", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            level, parsed = 0, {"reason": f"judge 输出解析失败: {raw[:80]}"}
        level = max(0, min(2, level))
        return {
            "level": level,
            "sufficient": level >= config.JUDGE_RELEVANCE_THRESHOLD,
            "reason": parsed.get("reason", ""),
            "usage": {
                "in": resp.usage.prompt_tokens,
                "out": resp.usage.completion_tokens,
            },
        }


def get_judge(threshold: float = 0.0) -> RuleBasedJudge | LLMJudge:
    """有 key 用 LLM judge，否则规则 judge。"""
    if config.LLM_API_KEY:
        return LLMJudge()
    return RuleBasedJudge(threshold)
