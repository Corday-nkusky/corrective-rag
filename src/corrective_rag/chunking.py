"""中文友好的递归字符切分器（Phase 1 组件，零第三方依赖）。

设计要点：
- 分隔符按优先级从粗到细：先按段落/整句切，切不动了才按逗号/空格，最后硬截断。
  英文世界的默认方案（\\n\\n + 空格）对中文基本无效：中文没有空格，
  按段落切完往往还是一整段超长文本。
- overlap 用字符数而非 token 数：本项目 embedding 是 BPE 字符级模型，
  字符重叠已足够近似语义重叠，且不需要引入 tokenizer 依赖。
- 不变量：任何情况下返回的每个 chunk 长度都 <= chunk_size
  （超过的被递归下放到更细的分隔符，最终硬截断收尾）。
  smoke 脚本对这个不变量做断言。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

DEFAULT_SEPARATORS: Sequence[str] = [
    "\n\n", "\n",
    "。", "！", "？",
    "；", ";",
    ".", "!", "?",
    "，", ",",
    " ",
]


def _split_keep_sep(text: str, sep: str) -> List[str]:
    """按 sep 切分但把分隔符保留在每段末尾，避免语义边界信息丢失。"""
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    return [p + sep for p in parts[:-1]] + [parts[-1]]


def _merge(pieces: Sequence[str], chunk_size: int, overlap: int) -> List[str]:
    """贪心合并碎片：能塞进当前块就塞，塞不下就结算，下一块带 overlap 尾巴起头。"""
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if not piece.strip():
            continue
        if len(piece) > chunk_size:
            # 单片就超长：先结算 current，再原样入队，
            # 由上层用更细的分隔符递归处理它。
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
            continue
        if not current:
            current = piece
        elif len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            chunks.append(current)
            current = (current[-overlap:] if overlap > 0 else "") + piece
    if current:
        chunks.append(current)
    return chunks


def _hard_cut(text: str, chunk_size: int, overlap: int) -> List[str]:
    """没有任何可用分隔符时的保底：按长度硬切，块间保留 overlap。"""
    step = max(1, chunk_size - overlap)
    return [text[start:start + chunk_size] for start in range(0, len(text), step)]


def recursive_split(
    text: str,
    chunk_size: int = 400,
    overlap: int = 60,
    separators: Optional[Sequence[str]] = None,
) -> List[str]:
    """把一段文本递归切成不超过 chunk_size 的块列表。

    参数
    -----
    text : 待切分文本
    chunk_size : 目标块长（字符数）
    overlap : 相邻块重叠字符数，必须 < chunk_size
    separators : 分隔符优先级列表，从粗到细
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap({overlap}) 必须小于 chunk_size({chunk_size})")
    if separators is None:
        separators = DEFAULT_SEPARATORS

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    # 找到第一个在当前文本中出现的分隔符
    sep = next((s for s in separators if s in text), None)
    if sep is None:
        return [c for c in (x.strip() for x in _hard_cut(text, chunk_size, overlap)) if c]

    merged = _merge(_split_keep_sep(text, sep), chunk_size, overlap)

    # 合并后仍超长的块，用更细一级的分隔符递归
    rest = list(separators)
    rest.remove(sep)
    result: List[str] = []
    for chunk in merged:
        if len(chunk) <= chunk_size:
            result.append(chunk)
        else:
            result.extend(recursive_split(chunk, chunk_size, overlap, rest))

    return [c for c in (x.strip() for x in result) if c]
