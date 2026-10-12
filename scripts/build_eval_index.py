"""把 DuReader-retrieval 的 100,001 条 passage 建为 FAISS 索引（Phase 2）。

与 smoke_pipeline 的区别：这次是真实规模数据（10 万段），跑在 GPU 上，
耗时分钟级；索引落 data/index/，评测脚本直接加载。

用法：
    set HF_ENDPOINT=https://hf-mirror.com   # data_loader 已下载则可离线
    PYTHONPATH=src python scripts/build_eval_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.data_loader import load_local  # noqa: E402
from corrective_rag.ingest import ingest_records  # noqa: E402


def main() -> None:
    data = load_local()
    records = [
        {"id": p["pid"], "source": "dureader", "text": p["text"]}
        for p in data["corpus"]
    ]
    print(f"passage 数: {len(records)}")
    ingest_records(records, config.INDEX_DIR)
    print(f"索引已写入: {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
