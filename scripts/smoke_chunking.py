"""Phase 0 冒烟测试：验证切分器不变量，不装任何第三方依赖也能跑。

用法：
    C:/Users/TANGWENJIAN/.workbuddy/binaries/python/versions/3.13.12/python.exe scripts/smoke_chunking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corrective_rag.chunking import recursive_split  # noqa: E402

SAMPLE = """操作系统引论

操作系统是配置在计算机硬件上的第一层软件，负责管理计算机系统的资源、进程和文件。没有操作系统的机器只能运行裸机程序，用户需要直接操作硬件寄存器，效率极低。

进程与线程

进程是资源分配的基本单位，线程是CPU调度的基本单位。同一进程内的多个线程共享地址空间，但每个线程有独立的栈。线程切换的开销远小于进程切换，因为不需要切换页表。

死锁

死锁产生的四个必要条件是：互斥条件、请求与保持条件、不剥夺条件和循环等待条件。只要破坏其中任意一个，死锁就不会发生。银行家算法通过安全性检测来避免死锁，它要求系统预先知道每个进程的最大资源需求。

虚拟内存

虚拟内存的核心思想是让每个进程拥有独立的地址空间，通过页表把虚拟地址映射到物理地址。当访问的页面不在内存中时，触发缺页中断，由操作系统负责把页面从磁盘换入。LRU 页面置换算法选择最久未使用的页面换出，Belady 现象表明 FIFO 算法在增加物理块数时缺页率反而可能上升。"""


def main() -> None:
    chunks = recursive_split(SAMPLE, chunk_size=120, overlap=30)

    print(f"输入长度: {len(SAMPLE)} 字符")
    print(f"chunk 数: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i:02d}] len={len(c):3d} | {c[:38].replace(chr(10), ' / ')}...")

    # 不变量 1：所有 chunk 不超过 chunk_size
    bad = [(i, len(c)) for i, c in enumerate(chunks) if len(c) > 120]
    assert not bad, f"存在超长 chunk: {bad}"

    # 不变量 2：无空 chunk
    assert all(c.strip() for c in chunks), "存在空 chunk"

    # 不变量 3：关键句子完整保留在某个 chunk 中
    key = "死锁产生的四个必要条件是：互斥条件、请求与保持条件、不剥夺条件和循环等待条件。"
    assert any(key in c for c in chunks), "关键句子被切断"

    # 不变量 4：内容基本无损（去空白后总字符数 >= 原文 95%，overlap 允许少量重复）
    joined_len = sum(len(c.replace(" ", "").replace("\n", "")) for c in chunks)
    src_len = len(SAMPLE.replace(" ", "").replace("\n", ""))
    coverage = joined_len / src_len
    print(f"覆盖率(含重叠): {coverage:.1%}")
    assert coverage >= 0.95, f"切分丢内容: {coverage:.1%}"

    print("\nSMOKE OK - chunking.py 全部断言通过")


if __name__ == "__main__":
    main()
