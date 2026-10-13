"""查询路由（Phase 6）：Adaptive-RAG 的 classifier，决定"要不要检索"。

论文血缘：Adaptive-RAG（arXiv:2403.14403，NAACL 2024）按查询复杂度把
query 分流到 不检索 / 单步检索 / 多步检索。落到本系统的两分类：

    direct   —— 闲聊、问候、稳定通用常识：模型参数化知识足以可靠回答，
                跳过检索/重排/judge，直接生成（省 rerank CPU + judge 一次调用）
    retrieve —— 需要具体事实/领域知识/本地语料覆盖：走完整纠错循环
                （检索→rerank→judge→重写回环/拒答）

为什么是两分类而不是论文的三分类：
- 论文的"多步检索"在本系统由纠错循环承担（迭代 + 重写本身就是多步机制），
  路由层只需要决定"要不要检索"；
- 论文的"单步检索"（judge 不过即弃）与我们的实测冲突：judge 误杀率
  16.9%（300 条一致率实测，FN 48 条），单趟就 abstain 会放大误杀，
  而第一轮重写正是这些 case 的主要兜底。检索路线统一走完整循环。

为什么路由在 condense 之后："那它呢"不结合历史连指代都消解不了，
分类更无从谈起。condense 空历史透传不花钱，router 看到的是已消解的问题。

两种实现，同一接口（有 key 用 LLM，没 key 用规则跑通链路）：
- LLMRouter：MiMo 结构化输出二分类，few-shot 稳定小模型行为
- RuleBasedRouter：问候/闲聊正则 → direct，其余一律 retrieve。
  刻意偏向 retrieve：误判 direct 的领域问题会得到无依据答案（幻觉），
  误判 retrieve 只是多花一次检索——不对称代价决定不对称默认
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402

ROUTER_PROMPT = """你是查询路由器。判断下面的用户问题应该走哪条处理路线。

路线定义：
- direct：闲聊、问候、表达感谢，或稳定的通用常识/定义/简单计算——模型自身的知识足以可靠回答，不需要外部语料。
- retrieve：需要具体事实、领域知识、本地语料覆盖的信息（如医疗、健康、生活、法律、时事等）——模型自身知识不可靠或可能过时，必须检索语料。

示例：
问题：你好呀 → {{"route": "direct", "reason": "问候闲聊"}}
问题：中国的首都是哪里 → {{"route": "direct", "reason": "稳定常识"}}
问题：3乘以7等于多少 → {{"route": "direct", "reason": "简单计算"}}
问题：感冒了吃什么药好 → {{"route": "retrieve", "reason": "医疗领域需查语料"}}
问题：风热感冒和风寒感冒怎么区分 → {{"route": "retrieve", "reason": "领域知识语料更可靠"}}

现在判断这个问题：
问题：{question}

只输出 JSON，不要输出任何其他内容：
{{"route": "direct或retrieve", "reason": "一句话理由，20字以内"}}"""


class RuleBasedRouter:
    """无 LLM 的规则路由：问候/闲聊模式命中即 direct，其余 retrieve。

    为什么不用"查询长度"当主要特征：中文领域问句可以很短
    （"感冒了怎么办" 6 字，必须检索），长度阈值在这个分布上不可分。
    论文里 query length 是强基线，但那是英文开放域 QA 的分布——
    换个分布就要重新校准，所以这里只保留最安全的模式匹配，
    默认全进 retrieve（宁慢勿错）。
    """

    # 命中任一模式 → direct。刻意保守：只收明确的问候/闲聊/身份类
    PATTERNS = [
        r"^(你好|您好|嗨|哈喽|喂|在吗|在么|早上好|中午好|下午好|晚上好|晚安|早安)",
        r"^(谢谢|感谢|多谢|辛苦了|麻烦你了|好的|行|可以|没事)",
        r"^(再见|拜拜|晚安|回头见)",
        r"^(你是谁|你叫什么|你是什么模型|你能做什么|你会什么|介绍一下自己)",
        r"^(hello|hi|hey|good\s*(morning|afternoon|evening)|thanks|thank\s*you|bye)",
    ]

    def route(self, question: str) -> Dict:
        q = question.strip()
        for pat in self.PATTERNS:
            if re.search(pat, q, re.IGNORECASE):
                return {
                    "route": "direct",
                    "reason": f"命中规则模式: {pat[:20]}",
                    "classifier": "rule",
                }
        return {"route": "retrieve", "reason": "未命中闲聊模式", "classifier": "rule"}


class LLMRouter:
    """MiMo 查询分类。一次轻量调用（短 prompt + json 输出），
    输出 direct/retrieve + 理由。失败时降级为 retrieve（安全侧）。"""

    def __init__(self) -> None:
        if not config.LLM_API_KEY:
            raise RuntimeError("LLMRouter 需要 LLM_API_KEY（或 DEEPSEEK_API_KEY）")

    def route(self, question: str) -> Dict:
        from openai import OpenAI

        from corrective_rag.cost import record

        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": ROUTER_PROMPT.format(question=question),
                }
            ],
            temperature=0,
            max_tokens=120,  # route + 20 字理由 + JSON 骨架
            response_format={"type": "json_object"},
        )
        record(resp.usage, config.LLM_MODEL)
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            route = str(parsed.get("route", "")).strip().lower()
        except (json.JSONDecodeError, ValueError, TypeError):
            route, parsed = "", {"reason": f"router 输出解析失败: {raw[:80]}"}
        # 解析失败或输出意外值 → retrieve（安全侧，同 RuleBasedRouter 的偏向）
        if route not in ("direct", "retrieve"):
            route = "retrieve"
        u = resp.usage
        return {
            "route": route,
            "reason": parsed.get("reason", ""),
            "classifier": "llm",
            "usage": {
                "in": getattr(u, "prompt_tokens", None) if u else None,
                "out": getattr(u, "completion_tokens", None) if u else None,
            },
        }


def get_router() -> RuleBasedRouter | LLMRouter:
    """有 key 用 LLM 路由，否则规则路由。"""
    if config.LLM_API_KEY:
        return LLMRouter()
    return RuleBasedRouter()
