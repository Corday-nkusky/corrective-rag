"""检索接口：query → 向量 → top-k chunk（Phase 1 组件）。

注意：本模块是"朴素检索"（单向量召回），Phase 2 会在它后面接
bge-reranker 做精选、再后面接 LLM judge 做充分性判断。
现在先把召回链路本身验通。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.embedding import Embedder  # noqa: E402
from corrective_rag.store import VectorStore  # noqa: E402


class Retriever:
    """加载一次索引，反复检索。"""

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        index_dir = Path(index_dir or config.INDEX_DIR)
        self.store, self.chunks = VectorStore.load(index_dir)
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """返回 [{id, score, source, text}, ...]，按相关度降序。"""
        top_k = top_k or config.TOP_K_RETRIEVE
        qvec = self.embedder.encode_query(query)
        hits = self.store.search(qvec, top_k)
        results = []
        for faiss_id, score in hits:
            chunk = self.chunks[faiss_id]
            results.append(
                {
                    "id": chunk["id"],
                    "score": round(score, 4),
                    "source": chunk["source"],
                    "text": chunk["text"],
                }
            )
        return results


class FixedRetriever:
    """注入式检索器：直接返回给定文档，适用于"文档已由基准给定"的场景（RGB）。

    为什么不把 RGB 文档灌进 FAISS 索引再正常检索：RGB 的评测设计假设外部
    文档由基准提供（含噪声/反事实构造），灌进索引会把"向量检索召回失败"
    混进门控测试，偏离基准设计意图，数字也没法和 RGB 论文基线对话。
    绕过 retrieve 之后，rerank 照常跑——噪声文档的重排表现本身就是测试点
    （rerank 该把真相关文档顶到前面）。
    """

    def __init__(self, docs: List[Dict]) -> None:
        # docs: [{text, role}] → 统一成 Retriever.retrieve 的输出形状
        self.docs = [
            {
                "id": f"fixed-{i}",
                "score": 1.0,
                "source": d.get("role", "fixed"),
                "text": d["text"],
            }
            for i, d in enumerate(docs)
        ]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        return self.docs[:top_k] if top_k else list(self.docs)
