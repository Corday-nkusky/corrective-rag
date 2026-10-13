"""BGE 中文向量模型封装（Phase 1 组件）。

设计要点：
- document 侧和 query 侧分开编码：bge 系列官方约定 query 要加指令前缀、
  document 不加。漏加前缀检索效果明显下降（模型卡 README 明确要求）。
- 输出归一化到单位长度：下游 FAISS 用内积索引（IndexFlatIP），
  归一化之后内积在数值上等于余弦相似度，top-k 排序口径统一。
- 单例加载：模型文件只读一次盘，同一进程内重复 encode 复用实例
  （bge-small 约 100MB，重复加载在冒烟脚本里很浪费）。
"""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from corrective_rag import config

_EMBEDDER: Optional["Embedder"] = None

# sentence-transformers 的 encode 不保证并发线程安全（内部 DataLoader /
# tokenizer 共享状态）；评测脚本用线程池并发压 API 时，GPU 前向串行化即可。
# 单次 encode 仅几十毫秒，锁不构成瓶颈——真正的耗时在网络端的 LLM 调用。
_EMBED_LOCK = threading.Lock()


class Embedder:
    """把文本变成 512 维（bge-small-zh）归一化向量的编码器。"""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        """document 侧编码：不加前缀，返回 (n, dim) 已归一化矩阵。"""
        with _EMBED_LOCK:
            return self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )

    def encode_query(self, query: str) -> np.ndarray:
        """query 侧编码：加 bge 指令前缀，返回 (dim,) 一维向量。"""
        with _EMBED_LOCK:
            vecs = self.model.encode(
                [config.BGE_QUERY_INSTRUCTION + query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vecs[0]


def get_embedder() -> Embedder:
    """进程内单例：避免冒烟/评测脚本里反复加载模型权重。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER
