"""公开评测集本地化：DuReader-retrieval + RGB → 项目 jsonl 格式。

来源一：DuReader-retrieval（C-MTEB 打包版，2026-10-12 实测，经 hf-mirror）
- queries: C-MTEB/DuRetrieval split=queries，字段 {id, text}
- corpus:  C-MTEB/DuRetrieval split=corpus，字段 {id, text}
- qrels:   C-MTEB/DuRetrieval-qrels split=dev，字段 {qid, pid, score}
  qrels 是**单独一个 repo**（MTEB 惯例），漏掉它就只拿到 query 和 passage 池、
  没有任何相关性标注，评测无从谈起。
  score 为二值 '1'：相关/不相关，没有 graded 相关性。因此检索指标用
  Recall@k / hit@k，不用 nDCG（二值 qrels 下 nDCG 会退化成 hit 类指标）。

来源二：RGB（arXiv:2309.01431，AAAI 2024，官方 repo chen700564/RGB，
2026-10-13 下载）中文子集三个文件：
- zh.json      300 条 噪声鲁棒/负例拒绝基础集：positive(相关文档) + negative(噪声文档)
- zh_int.json  100 条 信息整合：answer 拆两个要点，positive 是 [文档, 所属要点] 对
                        （官方字段拼写为 asnwer1，是 upstream typo，照抄不"修"）
- zh_fact.json 100 条 反事实：answer(真) + fakeanswer(假) + positive_wrong(3 篇伪事实)
下载坑：三个文件都是 **JSONL**（不是 JSON 数组，json.load 直接炸）；
raw.githubusercontent.com 对大文件超时，走 jsdelivr CDN 镜像可下。
输出统一格式：{id, subset, query, answer, docs: [{text, role}]}
role ∈ positive / negative / fake（zh_fact 的 positive_wrong）。

输出（data/eval/）：
- dureader_queries.jsonl  {qid, text}
- dureader_corpus.jsonl   {pid, text}
- dureader_qrels.jsonl    {qid, pid, score(int)}
- dureader_meta.json      条数统计 + 来源 + 下载时间
- rgb_zh.jsonl            上述统一格式（三个 subset 合并）
- rgb_meta.json           RGB 规模统计 + 字段映射说明

用法：
    python -m corrective_rag.data_loader                # DuReader（需 HF 镜像）
    python -m corrective_rag.data_loader --split-dev-limit 200
    python -m corrective_rag.data_loader --rgb          # 只做 RGB 本地化（不下载，读 rgb_raw/）
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

RGB_FILE = "rgb_zh.jsonl"
RGB_META_FILE = "rgb_meta.json"
RGB_RAW_DIR = "rgb_raw"  # data/eval/rgb_raw/{zh,zh_int,zh_fact}.json（JSONL）


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


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_text(x) -> str:
    """RGB 的文档字段在不同 subset 结构不同：str 或 list[str]（片段列表）。
    统一拼成一篇文档文本。"""
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x)


def localize_rgb() -> Dict:
    """把 rgb_raw/ 的三个 JSONL 转成统一格式 rgb_zh.jsonl。

    统一格式：
      {id, subset, query, answer(list[str]), answer_parts?, fake_answer?, docs: [{text, role}]}
    role: positive=真相关文档 / negative=噪声文档 / fake=伪事实文档(zh_fact)
    """
    raw_dir = config.EVAL_DIR / RGB_RAW_DIR
    rows: List[Dict] = []
    stats = {}

    # ---- zh：噪声鲁棒 + 负例拒绝基础集 ----
    for r in _read_jsonl(raw_dir / "zh.json"):
        docs = [{"text": _as_text(t), "role": "positive"} for t in r["positive"]]
        docs += [{"text": _as_text(t), "role": "negative"} for t in r["negative"]]
        rows.append(
            {
                "id": f"zh-{r['id']}",
                "subset": "zh",
                "query": r["query"],
                "answer": r["answer"],
                "docs": docs,
            }
        )
    stats["zh"] = len(rows)

    # ---- zh_int：信息整合 ----
    # 注意：数据里**没有**"文档→要点"映射字段（asnwer1/answer2 只是答案要点），
    # 官方 int 评测只看最终答案是否覆盖两个要点；positive 每篇是 str 或 list[str]。
    n_int = 0
    for r in _read_jsonl(raw_dir / "zh_int.json"):
        docs = [{"text": _as_text(t), "role": "positive"} for t in r["positive"]]
        docs += [{"text": _as_text(t), "role": "negative"} for t in r["negative"]]
        rows.append(
            {
                "id": f"zh_int-{r['id']}",
                "subset": "zh_int",
                "query": r["query"],
                "answer": r["answer"],
                "answer_parts": [r.get("asnwer1"), r.get("answer2")],
                "docs": docs,
            }
        )
        n_int += 1
    stats["zh_int"] = n_int

    # ---- zh_fact：反事实（positive_wrong = 说假答案的文档）----
    n_fact = 0
    for r in _read_jsonl(raw_dir / "zh_fact.json"):
        docs = [{"text": _as_text(t), "role": "positive"} for t in r["positive"]]
        docs += [{"text": _as_text(t), "role": "fake"} for t in r["positive_wrong"]]
        docs += [{"text": _as_text(t), "role": "negative"} for t in r["negative"]]
        rows.append(
            {
                "id": f"zh_fact-{r['id']}",
                "subset": "zh_fact",
                "query": r["query"],
                "answer": [r["answer"]] if isinstance(r["answer"], str) else r["answer"],
                "fake_answer": r["fakeanswer"],
                "docs": docs,
            }
        )
        n_fact += 1
    stats["zh_fact"] = n_fact

    _write_jsonl(config.EVAL_DIR / RGB_FILE, rows)
    meta = {
        "source": "github.com/chen700564/RGB（AAAI 2024，arXiv:2309.01431），jsdelivr CDN 镜像下载",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "counts": stats,
        "total": len(rows),
        "format_notes": "原始文件为 JSONL；zh_int 的 answer1 字段官方拼写为 asnwer1（upstream typo，照抄）",
    }
    with open(config.EVAL_DIR / RGB_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def load_rgb_local(subset: str | None = None) -> List[Dict]:
    """读回 rgb_zh.jsonl；subset 限定 'zh' / 'zh_int' / 'zh_fact' 之一。"""
    rows = _read_jsonl(config.EVAL_DIR / RGB_FILE)
    if subset:
        rows = [r for r in rows if r["subset"] == subset]
    return rows


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
    parser = argparse.ArgumentParser(description="本地化公开评测集")
    parser.add_argument("--split-dev-limit", type=int, default=None)
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="只做 RGB 本地化（读 data/eval/rgb_raw/，不联网）",
    )
    args = parser.parse_args()
    if args.rgb:
        localize_rgb()
    else:
        download(args.split_dev_limit)


if __name__ == "__main__":
    main()
