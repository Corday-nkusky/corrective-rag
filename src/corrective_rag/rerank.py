"""cross-encoder 重排（Phase 3 第一层质检）。

设计要点：
- bi-encoder（bge-small 检索）快但粗：query 和 document 分开编码，
  相关性只能靠向量相似度间接推断；cross-encoder 把 query-document
  拼成一句过一次模型，直接输出相关性 logit，准得多但慢
  （每个 pair 一次前向，无法预计算文档向量）。
- 所以分层：bi-encoder 先召回 top-20（快、可预计算索引），
  cross-encoder 重排取 top-4（准）。工业标准两段式 retrieve-and-rerank。
- bge-reranker-base 是 XLM-Roberta 结构的 cross-encoder，输出一个
  logit；直接按 logit 降序排（sigmoid 单调变换不改变排序）。
- 评测口径：重排前 Recall@4 vs 重排后 Recall@4，二值 qrels。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402

_RERANKER: Optional["Reranker"] = None

# CrossEncoder.predict 同样不保证线程安全；评测并发时前向串行化（见 embedding.py
# 同款说明：锁的粒度是 GPU 前向，网络端 LLM 调用仍并发）。
_RERANK_LOCK = threading.Lock()


class Reranker:
    """bge-reranker cross-encoder 封装。"""

    def __init__(self, model_name: str = config.RERANKER_MODEL) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def score(self, query: str, texts: List[str]) -> List[float]:
        """对 query 与每个 text 的相关性打分，返回与 texts 等长的分数列表。"""
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        with _RERANK_LOCK:
            logits = self.model.predict(pairs, show_progress_bar=False)
        return [float(x) for x in logits]

    def rerank(
        self, query: str, chunks: List[Dict], top_k: Optional[int] = None
    ) -> List[Dict]:
        """对 chunks 按相关性降序重排，返回前 top_k 个（带 rerank_score）。"""
        top_k = top_k or config.TOP_K_FINAL
        if not chunks:
            return []
        scores = self.score(query, [c["text"] for c in chunks])
        ranked = sorted(
            zip(scores, chunks), key=lambda sc: sc[0], reverse=True
        )
        out = []
        for s, c in ranked[:top_k]:
            item = dict(c)
            item["rerank_score"] = round(s, 4)
            out.append(item)
        return out


def get_reranker() -> Reranker:
    """进程内单例：cross-encoder 模型较大，避免重复加载。"""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER
