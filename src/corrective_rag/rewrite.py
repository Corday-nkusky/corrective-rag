"""查询重写（Phase 3 纠错策略池的第一招）。

重写是纠错循环里风险最高的一步：改写漂移了用户意图，比不改写更糟
（指代消解错一次，整个查询带偏）。所以两个实现都带自检：

- LLMRewriter（有 key）：指代消解 + 口语化→关键词化，MiMo 完成
- PRFRewriter（无 key 也能用）：伪相关反馈。用首轮检索 top 文档里的
  高频实词扩展原 query。经典 IR 技术（RM3 的简化版），1980 年代
  至今仍在用，论文依据充分。无 LLM 依赖。

自检（两个实现共用）：改写后 query 与原文的 Jaccard 重合度低于
MIN_OVERLAP 时判定"漂移过度"，放弃改写、用原 query 重试——
宁可原地重试，不要带偏。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402

MIN_OVERLAP = 0.2

# 中文常见虚词/标点，PRF 扩展词里要去掉（无停用词文件依赖）
STOPWORDS = set(
    "的 了 是 在 我 你 他 她 它 们 这 那 有 和 就 不 也 都 要 会 能 可以 什么 "
    "怎么 怎样 如何 为 与 及 对 把 被 从 到 于 吗 呢 吧 啊 呀 嗯 请 问 一下 "
    "一个 一些 这个 那个 这些 那些 他们 我们 你们 它们 自己 如果 虽然 但是 "
    "因为 所以 而且 或者 还是 已经 正在 可能 应该 需要 时候 地方 东西 事情 "
    "。，、；：？！""''（）()[]【】《》<>…—·-. "
)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class LLMRewriter:
    """LLM 重写：指代消解 + 更适合检索的表述。"""

    PROMPT = """把下面的用户问题改写成更适合搜索引擎检索的形式。要求：
1. 补全指代（如"它""那"要结合上下文猜出实体，没有上下文就保留原样）
2. 口语化表述改为关键词化表述
3. 不改变问题的实质意图
4. 只输出改写后的问题本身，不要任何解释

原问题：{question}"""

    def rewrite(self, question: str, retrieved: List[Dict] | None = None) -> Dict:
        from openai import OpenAI

        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "user", "content": self.PROMPT.format(question=question)}
            ],
            temperature=0.3,
            max_tokens=100,
        )
        new_q = (resp.choices[0].message.content or "").strip()
        return self._checked(question, new_q, strategy="llm")

    def _checked(self, question: str, new_q: str, strategy: str) -> Dict:
        if not new_q:
            return {
                "rewritten": question,
                "applied": False,
                "strategy": strategy,
                "reason": "改写结果为空，保留原 query",
            }
        overlap = _jaccard(question, new_q)
        if overlap < MIN_OVERLAP:
            return {
                "rewritten": question,
                "applied": False,
                "strategy": strategy,
                "reason": f"漂移过度(jaccard={overlap:.2f}<{MIN_OVERLAP})，保留原 query",
            }
        return {
            "rewritten": new_q,
            "applied": True,
            "strategy": strategy,
            "reason": f"jaccard={overlap:.2f}",
        }


class PRFRewriter:
    """伪相关反馈重写：用首轮 top 文档的高频实词扩展 query。

    流程：jieba 分词 → 去停用词和单字词 → 取前 top_docs 篇文档里
    出现频次最高的 n_terms 个词 → 拼到原 query 末尾。
    不替换原 query，只追加——原意图的词一个不动，漂移风险低。
    """

    def __init__(self, top_docs: int = 3, n_terms: int = 5) -> None:
        self.top_docs = top_docs
        self.n_terms = n_terms

    def _extract_terms(self, texts: List[str]) -> List[str]:
        import jieba

        freq: Counter = Counter()
        for t in texts:
            for w in jieba.lcut(t):
                w = w.strip()
                if len(w) >= 2 and w not in STOPWORDS:
                    freq[w] += 1
        return [w for w, _ in freq.most_common(self.n_terms)]

    def rewrite(self, question: str, retrieved: List[Dict] | None = None) -> Dict:
        if not retrieved:
            return {
                "rewritten": question,
                "applied": False,
                "strategy": "prf",
                "reason": "无检索结果可做反馈",
            }
        terms = self._extract_terms([r["text"] for r in retrieved[: self.top_docs]])
        if not terms:
            return {
                "rewritten": question,
                "applied": False,
                "strategy": "prf",
                "reason": "未提取到扩展词",
            }
        new_q = question + " " + " ".join(terms)
        return {
            "rewritten": new_q,
            "applied": True,
            "strategy": "prf",
            "reason": f"扩展词: {terms}",
        }


def get_rewriter():
    """有 key 用 LLM 重写，否则 PRF。"""
    if config.LLM_API_KEY:
        return LLMRewriter()
    return PRFRewriter()
