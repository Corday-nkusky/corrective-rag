"""Phase 1 全链路冒烟：建临时语料 → ingest → retrieve → generate(stub)。

不碰 data/corpus 和 data/index（那是 Phase 2 真实数据的领地），
全程用临时目录，跑完自动清理。

用法：
    C:/Users/TANGWENJIAN/anaconda3/envs/pytorch_env/python.exe scripts/smoke_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag import config  # noqa: E402
from corrective_rag.generate import generate  # noqa: E402
from corrective_rag.ingest import ingest  # noqa: E402
from corrective_rag.retrieve import Retriever  # noqa: E402

DOCS = {
    "os-memory.md": """操作系统与虚拟内存

进程是资源分配的基本单位，线程是CPU调度的基本单位。同一进程内的多个线程共享地址空间，但每个线程有独立的栈。线程切换的开销远小于进程切换，因为不需要切换页表。

虚拟内存的核心思想是让每个进程拥有独立的地址空间，通过页表把虚拟地址映射到物理地址。当访问的页面不在内存中时，触发缺页中断，由操作系统负责把页面从磁盘换入。

LRU 页面置换算法选择最久未使用的页面换出。Belady 现象表明 FIFO 算法在增加物理块数时，缺页率反而可能上升，这说明 FIFO 不是一种栈式算法。
""",
    "os-deadlock.md": """死锁与银行家算法

死锁产生的四个必要条件是：互斥条件、请求与保持条件、不剥夺条件和循环等待条件。只要破坏其中任意一个，死锁就不会发生。

银行家算法通过安全性检测来避免死锁，它要求系统预先知道每个进程的最大资源需求。每次资源请求后，系统都会检查是否存在一个安全序列：能让所有进程按某种顺序逐个完成。
""",
    "gpu-notes.md": """显卡选购笔记

RTX 3070 采用 Ampere 架构，8GB 显存，适合 1080p 高画质游戏。显存带宽是 448GB/s，位宽 256bit。

对于大语言模型本地推理，显存容量往往比算力更关键：7B 参数的模型以 FP16 存储需要约 14GB 显存，量化到 INT8 后约 7GB，INT4 约 4GB。
""",
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = Path(tmp) / "corpus"
        index_dir = Path(tmp) / "index"
        corpus_dir.mkdir(parents=True)
        for name, text in DOCS.items():
            (corpus_dir / name).write_text(text, encoding="utf-8")

        print(">>> ingest")
        stats = ingest(corpus_dir, index_dir)
        assert stats["chunks"] > 0, "没有产出任何 chunk"

        print("\n>>> retrieve")
        retriever = Retriever(index_dir)
        print(f"索引 chunk 数: {len(retriever.chunks)}")

        # 用例 1：语料里有的问题，top1 应该来自对应文档
        q1 = "死锁产生的四个必要条件是什么？"
        r1 = retriever.retrieve(q1, top_k=3)
        print(f"\n[{q1}]")
        for r in r1:
            print(f"  score={r['score']:.4f} src={r['source']} | {r['text'][:40]}...")
        assert "os-deadlock.md" == r1[0]["source"], (
            f"top1 来源错误: {r1[0]['source']}"
        )
        assert r1[0]["score"] > 0.5, f"top1 分数过低: {r1[0]['score']}"

        # 用例 2：语料里没有的问题，分数应明显更低
        q2 = "南开大学哪个食堂最好吃？"
        r2 = retriever.retrieve(q2, top_k=3)
        print(f"\n[{q2}]")
        for r in r2:
            print(f"  score={r['score']:.4f} src={r['source']} | {r['text'][:40]}...")
        assert r2[0]["score"] < r1[0]["score"], "无关问题分数反而更高，检索可疑"

        print("\n>>> generate (stub)")
        answer = generate(q1, r1[: config.TOP_K_FINAL])
        assert answer.startswith("[stub]")

        print("\nSMOKE OK - Phase 1 全链路通过")


if __name__ == "__main__":
    main()
