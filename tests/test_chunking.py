"""pytest 单测：装完环境后跑 `python -m pytest tests/ -v`。

冒烟脚本（scripts/smoke_chunking.py）覆盖 happy path，
这里补边界情况。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag.chunking import recursive_split  # noqa: E402


def test_short_text_returned_as_is():
    assert recursive_split("短文本。", chunk_size=100, overlap=10) == ["短文本。"]


def test_empty_text():
    assert recursive_split("   \n  ", chunk_size=100, overlap=10) == []


def test_invalid_overlap_raises():
    import pytest

    with pytest.raises(ValueError):
        recursive_split("测试" * 200, chunk_size=100, overlap=100)


def test_all_chunks_within_size():
    text = "这是一段没有句号的超长文本" * 100  # 无任何分隔符 -> 走硬截断
    chunks = recursive_split(text, chunk_size=200, overlap=40)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) >= 5


def test_no_content_loss_on_normal_text():
    text = ("第一段内容在这里，包含逗号和句号。" * 5) + "\n\n" + ("第二段是全新的内容。" * 5)
    chunks = recursive_split(text, chunk_size=150, overlap=30)
    joined = "".join(chunks)
    # overlap 会造成少量重复字符，但不应该丢字
    assert len(joined) >= len(text) * 0.95
