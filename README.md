# corrective-rag

带检索质量门控的纠错式 RAG 对话系统（本地语料 + 国产 LLM API + CPU/单卡可跑）。

核心一句话：**检索结果不过关就不许生成**——先过 reranker 过滤相关性，再过 LLM judge
判断"够不够回答问题"，不合格走策略池（查询重写 / 切混合检索 / 拒答），最多重试
`MAX_ITERATIONS` 次，仍不合格则诚实拒答。

论文血缘：CRAG（arXiv:2401.15884）+ Adaptive-RAG（arXiv:2403.14403）。
不自称 "Agentic RAG"——那是伞术语（综述 arXiv:2501.09136），面试被追问定义会失分。

## 当前状态（2026-10-12）

- [x] Phase 0 环境与骨架：pytorch_env 依赖装齐并锁定（requirements.lock，233 包）、
      bge-small-zh 经 hf-mirror 下载验证、切分器冒烟通过、git 初始化
- [ ] Phase 1-6：见 `docs/01-项目计划书.md`

## 快速开始

```bash
# 1. 切到已有环境（实测：py3.10.18 + torch 2.6.0+cu124 + CUDA True）
conda activate pytorch_env

# 2. 依赖已在 requirements.lock 冻结；补装走 tuna 镜像：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>

# 3. 配环境变量
cp .env.example .env   # 然后编辑 .env 填入 DEEPSEEK_API_KEY

# 4. 冒烟测试
python scripts/smoke_chunking.py
```

注意：`ragas` 不在依赖里（import 即炸，原因见计划书坑清单 #13），四个评测指标为自写
judge 脚本实现，定义对齐 ragas 公开文档。

完整实施计划（分阶段、验收标准、评测集设计、指标定义、坑清单）：
**`docs/01-项目计划书.md`**。

## 目录

```
src/corrective_rag/   源码（config / chunking 已实现，其余按计划书分阶段落地）
data/corpus/          演示语料（默认用 DuRetrieval 基准自带 passage pool；个人 md 可选）
data/eval/            公开基准本地化缓存（DuRetrieval / RGB → jsonl）
data/index/           FAISS 索引（产物，git 不跟踪）
data/traces/          每次运行的 JSONL trace（所有简历数字的唯一出处）
scripts/              冒烟测试与工具脚本
tests/                pytest
requirements.lock     依赖锁定（2026-10-12 冻结）
```
