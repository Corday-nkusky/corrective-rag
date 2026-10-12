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
