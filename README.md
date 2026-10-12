# corrective-rag

带检索质量门控的纠错式 RAG 对话系统（本地语料 + 国产 LLM API + CPU/单卡可跑）。

核心一句话：**检索结果不过关就不许生成**——先过 reranker 过滤相关性，再过 LLM judge
判断"够不够回答问题"，不合格走策略池（查询重写 / 切混合检索 / 拒答），最多重试
`MAX_ITERATIONS` 次，仍不合格则诚实拒答。

论文血缘：CRAG（arXiv:2401.15884）+ Adaptive-RAG（arXiv:2403.14403）。
不自称 "Agentic RAG"——那是伞术语（综述 arXiv:2501.09136），面试被追问定义会失分。

## 当前状态（2026-10-13）

- [x] Phase 0 环境与骨架：pytorch_env 依赖装齐并锁定（requirements.lock，233 包）、
      bge-small-zh 经 hf-mirror 下载验证、切分器冒烟通过、git 初始化
- [x] Phase 1 朴素 RAG 全链路：`embedding.py`（bge 编码，query 前缀+归一化）→
      `store.py`（FAISS IndexFlatIP）→ `ingest.py`（摄取）→ `retrieve.py`（召回）→
      `generate.py`（MiMo 生成，无 key 时 stub 模式）。
      全链路冒烟通过（`scripts/smoke_pipeline.py`，临时目录不碰真实数据）
- [x] Phase 2 评测集与基线：DuReader-retrieval 本地化（`data_loader.py`，经 hf-mirror
      实测可用；2,000 条人工标注 query / 100,001 passage / 9,839 条二值 qrels，
      id 对账零差错）；100K passage 建索引（`scripts/build_eval_index.py`）；
      基线评测（`scripts/eval_retrieval.py`）：**Recall@10=80.5%、MRR@10=87.8%、
      hit@1=82.6%**，2,000 条 27 秒跑完。注意：绝对值可能含数据污染
      （bge 2023-08 晚于基准 2022），A/B 的 delta 才可信
- [x] Phase 3 纠错循环（代码完成，评测数字部分完成）：
      - 状态机 `pipeline.py`：retrieve → rerank → judge →（够）生成 /（不够）重写回环 /
        拒答终态，LangGraph 编译，mermaid 可导出
      - reranker 实测增益（500 条）：**Recall@4 60.1% → 68.4%**，救回 21 条、弄坏 0 条
      - 校准实测发现：**top1 rerank 分数饱和无区分度**（失败组 p90 > 0.99，与成功组
        完全重叠）——"用 reranker 分数当门控"这条路走不通的实证，也是 LLM judge
        必须判"充分性"而非"相关性"的理由
      - LLM judge 接小米 MiMo（mimo-v2.6-flash，OpenAI 兼容协议，json 结构化输出
        实测可用）；`cost.py` 成本核算，每 query 花费进 trace
      - 20 条试水：闭环 recall@4 **64.5% → 79.1%**、hit@4 **90% → 100%**，
        LLM judge 触发 2 次重写（规则 judge 版 0 次）
      - **300 条全量 A/B**（`scripts/eval_corrective.py`，trace: corrective_ab_20261013_022855.json）：
        recall@4 **59.1% → 71.9%**（+12.8pp）、hit@4 **88.7% → 98.6%**（+9.8pp）；
        abstain 率 **8.0%**、平均迭代 0.45、重写生效 0.38/query——judge 真实参与决策；
        709 次调用共 **¥1.10**，每 query ¥0.00368。闭环 recall 为 answered-only 口径
        （分母剔除 24 条拒答）；全量口径 hit@4 ≈ 90.7%
      - judge 与 DuReader 人工标注一致率（300 条，trace: judge_agreement_20261013_025826.json）：
        **agreement 82.7%、precision 98.3%**（judge 放行的 240 条中仅 4 条无资料——
        幻觉几乎漏不出去）；FN 48 条（16.9% 误杀，多数被重写兜底救回，A/B 的
        hit@4 98.6% 即来源）；level 分布 2:208 / 1:32 / 0:60；成本 ¥0.42。
        口径边界：人工真值是"top4 命中"而非"资料充分"，为代理指标
- [x] Phase 4 RGB 四场景评测（`scripts/eval_rgb.py --scenario all`，trace: rgb_*_20261013_*.json）：
      - 负例拒绝（全噪声输入，300 条）：**abstain 29.0%、幻觉率 53.0%**——明确弱点，
        判"够"的门控对同主题干扰文档大量失效（RGB 的 negative 是相关但错误的文档）
      - 噪声鲁棒（60% 噪声，300 条）：**accuracy 90.7%**——强项
      - 信息整合（100 条）：两要点覆盖 73.0%、至少覆盖一个 92.0%
      - 反事实（100 条）：**被骗率 66.0%、答对真答案仅 16.0%**——最弱环节，
        reranker/judge 挡不住"语义相关但事实错误"的文档（CRAG 用 web search
        兜底正是为此）
      - 四场景成本合计 ¥6.47（预估 ¥0.9 严重偏低：3 条试跑未触发迭代不具代表性，
        counterfactual 平均迭代 1.45 次）。MiMo 总花费约 ¥8.2/¥10 预算
- [ ] Phase 5-6：对话版、demo。见 `docs/01-项目计划书.md`

## 快速开始

```bash
# 1. 切到已有环境（实测：py3.10.18 + torch 2.6.0+cu124 + CUDA True）
conda activate pytorch_env

# 2. 依赖已在 requirements.lock 冻结；补装走 tuna 镜像：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>

# 3. 配环境变量
cp .env.example .env   # 然后编辑 .env 填入 LLM_API_KEY（当前用小米 MiMo）

# 4. 冒烟测试
python scripts/smoke_chunking.py    # 切分器不变量（零依赖）
python scripts/smoke_pipeline.py    # 检索+生成全链路（临时语料，需装依赖）
```

注意：`ragas` 不在依赖里（import 即炸，原因见计划书坑清单 #13），四个评测指标为自写
judge 脚本实现，定义对齐 ragas 公开文档。

完整实施计划（分阶段、验收标准、评测集设计、指标定义、坑清单）：
**`docs/01-项目计划书.md`**。

## 目录

```
src/corrective_rag/   源码（config / chunking / embedding / store / ingest / retrieve / generate 已实现）
data/corpus/          演示语料（默认用 DuRetrieval 基准自带 passage pool；个人 md 可选）
data/eval/            公开基准本地化缓存（DuRetrieval / RGB → jsonl）
data/index/           FAISS 索引（产物，git 不跟踪）
data/traces/          每次运行的 JSONL trace（所有简历数字的唯一出处）
scripts/              冒烟测试与工具脚本
tests/                pytest
requirements.lock     依赖锁定（2026-10-12 冻结）
```
