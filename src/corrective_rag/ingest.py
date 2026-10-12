"""语料摄取：扫描 → 切块 → 向量化 → 建索引 → 落盘（Phase 1 组件）。

用法：
    python -m corrective_rag.ingest                    # 用 config.CORPUS_DIR
    python -m corrective_rag.ingest --corpus X --index-dir Y

流程：遍历语料目录（.md/.txt/.markdown）→ recursive_split 切块 →
每个 chunk 记录 {id, source, text} → encode_docs 批量向量化 →
VectorStore 落盘。幂等：重复 ingest 会整体重建索引，不做增量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.chunking import recursive_split  # noqa: E402
from corrective_rag.embedding import Embedder  # noqa: E402
from corrective_rag.store import VectorStore  # noqa: E402

SUPPORTED_SUFFIXES = {".md", ".txt", ".markdown"}


def load_corpus_files(corpus_dir: Path) -> List[Path]:
    files = [
        p for p in sorted(corpus_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def build_chunks(corpus_dir: Path) -> List[Dict]:
    """把语料目录切成带元数据的 chunk 列表。"""
    chunks: List[Dict] = []
    for path in load_corpus_files(corpus_dir):
        text = path.read_text(encoding="utf-8", errors="replace")
        pieces = recursive_split(
            text, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP
        )
        for piece in pieces:
            chunks.append(
                {
                    "id": len(chunks),
                    "source": str(path.relative_to(corpus_dir)),
                    "text": piece,
                    "n_chars": len(piece),
                }
            )
    return chunks


def ingest(
    corpus_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
) -> Dict:
    corpus_dir = Path(corpus_dir or config.CORPUS_DIR)
    index_dir = Path(index_dir or config.INDEX_DIR)

    chunks = build_chunks(corpus_dir)
    if not chunks:
        raise RuntimeError(f"语料目录为空或无支持的文件: {corpus_dir}")

    embedder = Embedder()
    vectors = embedder.encode_docs([c["text"] for c in chunks])

    store = VectorStore(dim=embedder.dim)
    store.add(vectors)
    store.save(index_dir, chunks)

    stats = {
        "corpus_dir": str(corpus_dir),
        "index_dir": str(index_dir),
        "files": len(load_corpus_files(corpus_dir)),
        "chunks": len(chunks),
        "dim": embedder.dim,
        "avg_chars": round(sum(c["n_chars"] for c in chunks) / len(chunks), 1),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def ingest_records(
    records: List[Dict],
    index_dir: Optional[Path] = None,
) -> Dict:
    """直接对记录建索引，不切分——评测语料（DuReader passage）入口。

    与 ingest() 的区别：DuReader 的 passage 本身就是段落级单元
    （原论文平均 285 字符），且 Recall@k 的人工标注以 pid 为单位。
    再走 recursive_split 会破坏 pid 与 chunk 的一一对应，评测时
    检索命中的 chunk 无法映射回标注 id，Recall 无从对账。

    records 每项: {"id": <唯一id>, "source": <来源名>, "text": <正文>}
    """
    index_dir = Path(index_dir or config.INDEX_DIR)
    if not records:
        raise RuntimeError("records 为空，拒绝建索引")

    chunks = [
        {
            "id": r["id"],
            "source": r.get("source", "eval"),
            "text": r["text"],
            "n_chars": len(r["text"]),
        }
        for r in records
    ]
    ids = [c["id"] for c in chunks]
    assert len(set(ids)) == len(ids), "records 存在重复 id，落盘会错位"

    embedder = Embedder()
    vectors = embedder.encode_docs([c["text"] for c in chunks])

    store = VectorStore(dim=embedder.dim)
    store.add(vectors)
    store.save(index_dir, chunks)

    stats = {
        "index_dir": str(index_dir),
        "mode": "records(no-split)",
        "chunks": len(chunks),
        "dim": embedder.dim,
        "avg_chars": round(sum(c["n_chars"] for c in chunks) / len(chunks), 1),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="摄取语料并构建 FAISS 索引")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    args = parser.parse_args()
    ingest(args.corpus, args.index_dir)


if __name__ == "__main__":
    main()
