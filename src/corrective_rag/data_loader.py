"""公开评测集本地化：DuReader-retrieval（C-MTEB 打包版）→ 项目 jsonl 格式。

来源（2026-10-12 实测，经 hf-mirror）：
- queries: C-MTEB/DuRetrieval split=queries，字段 {id, text}
- corpus:  C-MTEB/DuRetrieval split=corpus，字段 {id, text}
- qrels:   C-MTEB/DuRetrieval-qrels split=dev，字段 {qid, pid, score}
  qrels 是**单独一个 repo**（MTEB 惯例），漏掉它就只拿到 query 和 passage 池、
  没有任何相关性标注，评测无从谈起。
  score 为二值 '1'：相关/不相关，没有 graded 相关性。因此检索指标用
  Recall@k / hit@k，不用 nDCG（二值 qrels 下 nDCG 会退化成 hit 类指标）。

输出（data/eval/）：
- dureader_queries.jsonl  {qid, text}
- dureader_corpus.jsonl   {pid, text}
- dureader_qrels.jsonl    {qid, pid, score(int)}
- dureader_meta.json      条数统计 + 来源 + 下载时间

用法：
    python -m corrective_rag.data_loader
    python -m corrective_rag.data_loader --split-dev-limit 200   # 只取前 N 条调参
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402

MAIN_REPO = "C-MTEB/DuRetrieval"
QRELS_REPO = "C-MTEB/DuRetrieval-qrels"

QUERIES_FILE = "dureader_queries.jsonl"
CORPUS_FILE = "dureader_corpus.jsonl"
QRELS_FILE = "dureader_qrels.jsonl"
META_FILE = "dureader_meta.json"


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def download(split_dev_limit: int | None = None) -> Dict:
    """下载三部分并落盘，返回统计信息。"""
    from datasets import load_dataset

    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)

    queries_ds = load_dataset(MAIN_REPO, split="queries")
    corpus_ds = load_dataset(MAIN_REPO, split="corpus")
    qrels_ds = load_dataset(QRELS_REPO, split="dev")

    queries = [{"qid": r["id"], "text": r["text"]} for r in queries_ds]
    corpus = [{"pid": r["id"], "text": r["text"]} for r in corpus_ds]
    qrels = [
        {"qid": r["qid"], "pid": r["pid"], "score": int(r["score"])} for r in qrels_ds
    ]

    # 只保留有 qrels 标注的 query：无标注 query 无法算 Recall，不占评测集
    labeled_qids = {q["qid"] for q in qrels}
    queries_all = queries
    queries = [q for q in queries if q["qid"] in labeled_qids]

    if split_dev_limit is not None and len(queries) > split_dev_limit:
        keep = set(q["qid"] for q in queries[:split_dev_limit])
        queries = queries[:split_dev_limit]
        qrels = [q for q in qrels if q["qid"] in keep]

    _write_jsonl(config.EVAL_DIR / QUERIES_FILE, queries)
    _write_jsonl(config.EVAL_DIR / CORPUS_FILE, corpus)
    _write_jsonl(config.EVAL_DIR / QRELS_FILE, qrels)

    # 只保留被标注 query 引用过的 corpus 会破坏"召回失败"场景的构造，
    # corpus 保持全量（无标注 query 的 passage 是干扰项，本身是评测的一部分）。
    stats = {
        "source": {"main": MAIN_REPO, "qrels": QRELS_REPO},
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "queries_total_in_repo": len(queries_all),
        "queries_labeled": len(queries),
        "corpus": len(corpus),
        "qrels_rows": len(qrels),
        "avg_relevant_per_query": round(len(qrels) / max(1, len(queries)), 2),
        "score_values": sorted({q["score"] for q in qrels}),
        "dev_limit": split_dev_limit,
    }
    with open(config.EVAL_DIR / META_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def load_local() -> Dict[str, List[Dict]]:
    """从 data/eval/ 读回三部分（评测脚本统一入口）。"""
    out: Dict[str, List[Dict]] = {}
    for key, fname in [
        ("queries", QUERIES_FILE),
        ("corpus", CORPUS_FILE),
        ("qrels", QRELS_FILE),
    ]:
        rows = []
        with open(config.EVAL_DIR / fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        out[key] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="本地化 DuReader-retrieval 评测集")
    parser.add_argument("--split-dev-limit", type=int, default=None)
    args = parser.parse_args()
    download(args.split_dev_limit)


if __name__ == "__main__":
    main()
