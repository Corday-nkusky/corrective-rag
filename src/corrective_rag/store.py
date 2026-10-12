"""FAISS 向量索引封装（Phase 1 组件）。

设计要点：
- IndexFlatIP（内积）：向量在 encode 时已归一化，内积即余弦相似度。
  暴力精确检索，不做 ANN 近似——本机语料规模（万级 chunk）下
  top-k 耗时在毫秒级，没必要为省几毫秒引入近似误差（面试也讲得清：
  "为什么不用 HNSW：规模不到，精确优先"）。
- 索引与文本分离落盘：index.faiss 只存向量，chunks.jsonl 存文本与来源，
  两文件第 i 条严格一一对应（faiss 的 add 顺序 = jsonl 行序）。
  分离的好处：换 embedding 模型不用重切语料，文本侧不用动。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.jsonl"


class VectorStore:
    """FAISS 索引 + chunk 元数据的一增一存一取。"""

    def __init__(self, dim: int) -> None:
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        assert vectors.shape[1] == self.dim, "向量维度与索引不一致"
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """返回 [(faiss_id, score), ...]，按分数降序（faiss 保证）。"""
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        scores, ids = self.index.search(q, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, index_dir: Path, chunks: List[Dict]) -> None:
        import faiss

        index_dir.mkdir(parents=True, exist_ok=True)
        assert len(chunks) == self.index.ntotal, (
            f"chunk 数({len(chunks)})与索引向量数({self.index.ntotal})不一致，"
            "落盘会错位，拒绝保存"
        )
        faiss.write_index(self.index, str(index_dir / INDEX_FILE))
        with open(index_dir / CHUNKS_FILE, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    @staticmethod
    def load(index_dir: Path) -> Tuple["VectorStore", List[Dict]]:
        import faiss

        store = VectorStore.__new__(VectorStore)
        store.index = faiss.read_index(str(index_dir / INDEX_FILE))
        store.dim = store.index.d
        chunks: List[Dict] = []
        with open(index_dir / CHUNKS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        if len(chunks) != store.index.ntotal:
            raise RuntimeError(
                f"索引与元数据错位: ntotal={store.index.ntotal}, jsonl={len(chunks)}"
            )
        return store, chunks
