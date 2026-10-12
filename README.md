# corrective-rag

带检索质量门控的纠错式 RAG 对话系统（本地语料 + 国产 LLM API + CPU/单卡可跑）。

核心一句话：**检索结果不过关就不许生成**——先过 reranker 过滤相关性，再过 LLM judge
判断"够不够回答问题"，不合格走策略池（查询重写 / 切混合检索 / 拒答），最多重试
`MAX_ITERATIONS` 次，仍不合格则诚实拒答。

论文血缘：CRAG（arXiv:2401.15884）+ Adaptive-RAG（arXiv:2403.14403）。
不自称 "Agentic RAG"——那是伞术语（综述 arXiv:2501.09136），面试被追问定义会失分。

## 当前状态（2026-10-12）

- [x] Phase 0 骨架：目录、配置、切分器（已跑通冒烟测试）
- [ ] Phase 1-6：见 `docs/01-项目计划书.md`

## 快速开始

```bash
# 1. 建虚拟环境（用 managed Python 3.13.12）
C:/Users/TANGWENJIAN/.workbuddy/binaries/python/versions/3.13.12/python.exe -m venv .venv
source .venv/Scripts/activate

# 2. 装依赖（约 2-3GB，主要是 torch 的 CUDA wheel）
pip install -r requirements.txt
pip freeze > requirements.lock

# 3. 配环境变量
cp .env.example .env   # 然后编辑 .env 填入 DEEPSEEK_API_KEY

# 4. 冒烟测试（不装任何第三方依赖也能先跑）
python scripts/smoke_chunking.py
```

完整实施计划（分阶段、验收标准、评测集设计、指标定义、坑清单）：
**`docs/01-项目计划书.md`**。

## 目录

```
src/corrective_rag/   源码（config / chunking 已实现，其余按计划书分阶段落地）
data/corpus/          中文语料（.md/.txt，你亲手标过答案的领域）
data/eval/            评测集（jsonl：query / 标准答案 / 支撑 chunk id / 类型标签）
data/index/           FAISS 索引（产物，git 不跟踪）
data/traces/          每次运行的 JSONL trace（所有简历数字的唯一出处）
scripts/              冒烟测试与工具脚本
tests/                pytest
```
