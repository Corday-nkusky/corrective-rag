# 02 · Agentic RAG 前沿调研（2026-10-13）

> 定位：本文回答"我们在宇宙哪里"——项目与 2026 年前沿实践的位置关系。与 00（全景/设计意图）、01（计划书/执行）并列，本文管**外部坐标**。
> 事实分级约定：【官方】= 论文/官方仓库/官方榜单原文；【实测】= 本项目 trace 实跑；【第三方】= 博客/厂商/聚合器转述，未独立复现；【未验证】= 训练数据记忆或单一不可溯来源。文末附未验证清单。

---

## 0. 结论先行

1. **"Agentic RAG"在 2026 年已是有综述定义的成熟术语，不是营销词。** Singh et al. 的 survey（arXiv:2501.09136，v4 修订于 2026-04-01）给出五范式演进 + 七种 agentic 架构的分类学，是社区引用的事实标准地图【官方 arXiv】。本项目实现的 CRAG + Adaptive-RAG 正是七种架构中的两种。
2. **2026 前沿主线不是"更花的编排"，而是三条**：① 检索策略从 prompt 工程走向可学习策略（Search-R1 → ZeroSearch → SPARKLE），且主流做法是把策略放在主模型**外面**；② 记忆层从"查文档"扩展到"记经验"（Mem0 / Continuum Memory / MAGMA）；③ 深度研究 agent 把整套系统推到开放 web（BrowseComp 系基准 + Tongyi/Kimi 开源系统）。
3. **工程共识已固化**：混合检索 + RRF + cross-encoder 精排是底座；"要不要检索"的路由门是 2026 生产系统 ROI 最高、最被低估的组件；LangGraph 是默认编排；MCP 是检索的**控制平面**而非检索算法；"上下文工程"成为总名词，RAG 降级为它的一个子系统。
4. **冷静数据同样重要**：GraphRAG 九方法横评中最好的也只比"不检索的 GPT-4o-mini 基线"高 1.9pp；AgenticRAGTracer 上 GPT-5 最难多跳仅 22.6% EM——多跳**规划**（不是检索）是行业瓶颈；agentic RAG 的 token 成本是简单 RAG 的 3-10 倍。
5. **对本项目**：router（Adaptive-RAG 二分类）已于 2026-10-13 实现并实测（准确率 96.7%，直答路线快 2.4 倍 / 便宜 6 倍）【实测】——恰好落在 2026 工程共识里 ROI 最高的一块（retrieve-or-skip 路由门）。剩下的大项只有 test split 一次终测；dev/test 分离 + 污染披露的协议纪律优于行业平均（GAIA 答案公开、search-time contamination 都是 2026 显学）。

---

## 1. 概念谱系：这个术语到底指什么

### 1.1 五范式演进

survey（2501.09136）把 RAG 的演进分为五档【官方 arXiv 摘要确认五范式；架构细节经第三方解读 beri.net 转述】：

| 范式 | 一句话特征 | 死因 |
|---|---|---|
| Naive | 一次检索，top-k 塞 prompt，生成 | 检索错就整段垮，无纠错 |
| Advanced | 混合检索、精排、query 重写 | 流程写死，复杂查询仍不够 |
| Modular | 编排框架化，组件可插拔 | 组件间无决策，只是流水线 |
| Graph | 知识图谱承担多跳 | 构建贵、增量难、增益有限（见 2.2） |
| Agentic | agent 决定何时检索/检索什么/检索几次/何时停 | 成本与可靠性（见第 5 节） |

类比（408 视角）：Naive 是"grep 一次定生死"；Advanced 是"倒排 + 索引回表"；Agentic 是"写个调度程序，自己决定查几个库、查不到怎么办"。

### 1.2 七种 agentic 架构

【第三方对论文的解读，论文摘要确认了分类维度：agent 基数、控制结构、自主度、知识表示】

1. 单 agent 路由（single-agent router）——一个 agent 把 query 分派到不同管线
2. 多 agent 系统，各配专职检索器
3. 层级 agent（上级分解、下级执行）
4. **Corrective RAG**——相关性 grader 打分，低分则重写 query 重试/换源 ← **本项目已实现并实测**
5. **Adaptive RAG**——分类器按难度决定跳过检索 / 单跳 / 多跳 ← **本项目已实现**（`router.py`，两分类 direct/retrieve，2026-10-13 实测准确率 96.7%）
6. 图增强 agentic（Agent-G、GeAR 框架）
7. agentic 文档工作流（长文档解析-检索-生成一体化）

### 1.3 2026 的形式化：SoK 把 agentic RAG 建模成 POMDP

arXiv:2603.07379（2026-03-07 提交）【官方 arXiv 摘要，本轮已直核】：把"检索-生成回环"形式化为**有限时域部分可观察马尔可夫决策过程**（finite-horizon POMDP），显式建模控制策略与状态转移。指出的四类系统性风险：

- 幻觉复利传播（compounding hallucination propagation）——错误证据喂出错误前提，agent 在错误前提上继续建楼
- 记忆投毒（memory poisoning）
- 检索错位（retrieval misalignment）
- 工具执行级联漏洞（cascading tool-execution vulnerabilities）

研究方向：稳定自适应检索、成本感知编排、轨迹形式化评测、监督机制。

> 对本项目的直接意义：我们的 counterfactual 场景"被骗率 66.0%"【实测】就是幻觉复利传播的一个实例——fake 文档通过 judge 后进入生成， rerank/judge 都挡不住"语义相关但事实错误"。

---

## 2. 学术前沿四条线

### 2.1 线一：检索策略学习化（2025-2026 最密集的一条）

问题动机（SPARKLE 论文原话方向，ACL 2026【官方】）：现有方法要么靠冻结 LLM 无显式监督地自己决定（次优），要么付出高昂代价微调主 LLM（伤泛化）。

| 工作 | 做法 | 关键数字 | 分级 |
|---|---|---|---|
| Self-RAG（2023） | 训练 reflection token（[Retrieve]/[IsRel]/[IsSup]/[IsUse]） | 幻觉显著下降 | 官方论文，本轮未直核 |
| Search-R1（arXiv:2503.09516） | RL 让模型在推理中自主发搜索 query；retrieved token masking 稳定训练 | +41%（Qwen2.5-7B）/ +20%（3B）vs 同设置 RAG 基线 | 官方 arXiv 摘要 |
| R1-Searcher | 两阶段 outcome-based RL | — | 第三方转述 |
| ZeroSearch（arXiv:2505.04588，ICLR 2026） | 用 LLM **模拟**搜索引擎：SFT 学 snippet 风格 + 课程退化（有用→噪声）+ F1 奖励（防 EM reward hacking） | 7B 模拟器 ≈ 真 Google，14B 超越；API 成本归零 | 官方 arXiv + OpenReview 收录信息 |
| SPARKLE（ACL 2026 long） | **proxy model** 独立学检索策略（KG 结构化决策 + RL + 二叉树 rollout），不改 LLM 本体，即插即用 | in-domain +9.17% / OOD +2.85% | 官方 ACL Anthology |
| ReflectiveRAG（EACL 2026 industry） | 小模型当决策控制器迭代评估证据充分性 + 对比式噪声过滤 | +2.7pp EM / +2.5pp F1 over DeepRAG；冗余 -30.88%；仅 +18ms | 官方 ACL Anthology |
| KiRAG（ACL 2025） | 知识三元组驱动迭代检索 | R@3 +9.40% / F1 +5.14% | 官方 ACL Anthology |
| CorVer（arXiv 2026） | Infini-gram 语料共现当过程奖励 | 30 格全胜，平均 +4.1 EM（TriviaQA），比神经验证器便宜 4.8-8.4× | 第三方列表转述 |
| QuCo-RAG（ACL Findings 2026） | 用预训练语料统计（而非 logits/entropy）量化不确定性决定何时检索 | +5-12 EM over DRAGIN | 第三方列表转述 |

**趋势判读**：决策权从"写死的 if-else / prompt"转向"可学习的策略"；但 2026 的聪明解法是**把策略放在 LLM 外面**（proxy model / 小控制器），不动主模型——SPARKLE 的 plug-and-play 是代表，ReflectiveRAG 的"小模型决策器 + 不要策略训练"是另一条更轻的路。

> 与本项目的共鸣点：SPARKLE 的动机（冻结 LLM 无监督决策次优）正是我们实测过的现象——reranker top1 分数饱和无区分度（失败组 p90>0.99）、prompt 式 LLM judge 对人工 qrels 误杀率 16.9%【实测】。我们的负面实证与 2026 学术动机同向，这是面试里能讲的故事。

### 2.2 线二：图与多模态（增益比宣传小）

GraphRAG-Bench（arXiv:2506.02404）【官方 arXiv，九方法横评，20 本 CS 教科书构建的域特定推理基准】：

| 方法 | 平均准确率 | 索引时间 | 平均检索时间 |
|---|---|---|---|
| GPT-4o-mini（不用检索的基线） | 70.68% | — | — |
| RAPTOR | 73.58% | 451s | 0.02s |
| HippoRAG | 72.64% | 4695s | 2.44s |
| GraphRAG（MS） | 72.50% | 1797s | 44.87s |
| KGP | 71.86% | 204s | 89.38s |
| GFM-RAG | 72.10% | 94s | 1.96s |
| LightRAG | 71.22% | 1430s | 13.95s |
| DALK | 69.30%（**低于基线**） | 407s | 26.80s |
| G-Retriever | 69.84%（**低于基线**） | 920s | 23.77s |

冷静结论：图方法相对"不检索基线"最好只高 1.9pp，两个方法反而拖累基线；数学域下降（符号操作与检索信息不匹配）。图谱构建的 token 成本是普通 RAG 的 5-10 倍、文档一更新就要重建【第三方中文博客，与横评的索引时间数据互相印证】。

其他值得知道的名字：
- **KAG**（蚂蚁 OpenSPG）：知识抽取 + 逻辑符号引导的混合推理，面向事实问答 + 逻辑推理【第三方】
- **LightRAG**（HKUDS）：双层图 + 增量更新【第三方】
- **HCG-RAG**（BCG，2026-07）：schema 约束因果图，构建 LLM 调用比 MS GraphRAG 少 8-135×【第三方中文源，未核原文】
- **RAG-Anything**（HKUDS 黄超团队，2026-01 开源）：多模态统一 RAG，跨模态知识图谱，MinerU 解析，文字/图像/表格/公式统一问答【第三方中文源】

### 2.3 线三：Agent 记忆层（RAG 不够的新论点）

核心批评来自 Continuum Memory Architecture（arXiv:2601.09913，2026-01）【第三方转述，未直核原文】：RAG 把记忆当**无状态查找表**——只累积不更新、检索只读、无时间连续性；agent 没有"何时学的、现在还有效吗、记忆之间什么关系"的模型。提出五原语：持久存储、选择性保留、联想路由、时间链、巩固。

代表系统：
- **Mem0**（arXiv:2504.19413）：事实三元组存储 + LLM 抽取 + 整合去重（矛盾时更新而非堆叠）+ 图内存 Mem0g。官方口径：LOCOMO LLM-as-Judge 67.13%、比全量上下文省 token 93%、延迟 -91%、比 OpenAI Memory +26%；AWS Agent SDK 独家记忆提供商；融资 $24M【官方博客 + 第三方】
- **MAGMA**（2026-01 v1 / 2026-04 v2）：四张正交图（语义/时间/因果/实体）表示同一条记忆，策略引导遍历【第三方】
- **Zep / Graphiti**：LongMemEval 时间推理比竞品高 15 分【第三方】

生产记忆四层栈（第三方综合）：① 对话记忆（上下文窗口内）② 工作记忆（会话内中间结果外置）③ 长期记忆（跨会话，向量/图/混合）④ 共享记忆（多 agent 并发，竞态控制）。未解决问题：记忆陈旧（一月对的记忆三月有害）、隐私治理、跨会话身份解析。市场口径：$6.27B（2026）→ $28.45B（2030），CAGR 35%（Mordor Intelligence）【第三方】。

> 与本项目的关系：Phase 5 的 ConversationMemory（滑窗 3 轮 + 超窗 LLM 增量摘要）只到第 ①②层【实测】。跨会话结构化记忆是明确的下一步候选，但不是简历必需。

### 2.4 线四：深度研究 agent（把 agentic RAG 推到开放 web）

基准地图：

| 基准 | 测什么 | 注意 |
|---|---|---|
| BrowseComp（OpenAI，arXiv:2504.12516，2025-04） | 1,266 题开放 web 难检索事实，短答案 exact match | 人类 2h 限时基线 29.2%（论文口径，benchgen 转述 + 训练数据一致） |
| BrowseComp-Plus（arXiv:2508.06600） | 固定 100K 文档语料上的可复现版 | 消除"live web 漂移" |
| DeepResearch Bench II（arXiv:2601.08536） | 132 个 grounded 任务、9,430 条专家二元 rubric（RACE/FACT 评分） | 论文报告最强 agent 不到一半分；live 榜 2026-10-01 最高 64.38【第三方快照】 |
| GAIA | 466 题通用助手，三级难度 | 验证答案公开已污染；test >93% 接近饱和 |
| xbench-DeepSearch（清华 + 港大，2025） | 中英双语，多文档综合/比较/时序推理 | BrowseComp 只测单实体 |
| DRACO（arXiv:2602.11685） | 真实 deep-research 请求采样 | 第三方转述，未直核 |

SOTA（benchgen 汇总，2026-07-16，Inkling model card 口径，context management 开启）【第三方榜单】：GPT-5.6 Sol 90.8 / Claude Fable 5 88.0 / Gemini 3.1 Pro 85.9 / DeepSeek V4 Pro 83.4 / Kimi K2.6 83.2 / Inkling 77.1 / Kimi K2.5 74.9。**都已远超人类 2h 基线 29.2%**——这正是"检索 persistence 可被 test-time compute 堆出来"的证据。

开源深度研究系统：Tongyi DeepResearch（阿里，30B-A3B MoE：BrowseComp 43.4 / GAIA 70.9 / HLE 32.9 / xbench-DeepSearch 75.0，Heavy 模式做测试时并行综合）【第三方转述技术报告】；Kimi K2.5（月之暗面，2026-01-27，1T-A32B：BrowseComp 74.9 / HLE w/ tools 50.2 / DeepSearchQA 89.0）【官方发布转述 + 第三方】；DR Tulu / OpenResearcher / QUEST-35B / MiroThinker 等开源训练框架【第三方】。

评测教训（aievals.co【第三方】）：BrowseComp/GAIA 只测"检索 persistence"，模型可以拿满分但报告写得浅而差——主指标应是 RACE 式 rubric；citation accuracy ≠ factual correctness（FACT 只验证"来源支持该 claim"，不验证来源本身对）。

---

## 3. 工程实践：2026 生产栈

### 3.1 底座四件套（无争议部分）

1. **混合检索**：dense + BM25，RRF 融合。纯向量检索被明确视为架构错误【第三方多源一致】。
2. **Cross-encoder 精排**：Cohere Rerank 3.5 / Voyage rerank-2.5，粗排 top-20 → 精排 top-4-5【第三方】。与本项目 retrieve top20 → rerank top4 的结构一致【实测】。
3. **Query 重写**：HyDE（假设答案再 embedding）/ multi-query / step-back【第三方】。本项目的 rewrite 节点属于这一类【实测】。
4. **Observability**：Langfuse / Arize Phoenix，检索精度、faithfulness、成本全链路追踪【第三方】。本项目的 trace jsonl + cost.py 是同思路的轻量版【实测】。

### 3.2 路由门：最被低估的组件

retrieve-or-skip（先判断要不要检索）被称为"最简单却最被低估的模式"【第三方】。生产口径：60-70% 的 query 是简单的；智能分类可降本 30-45%、降延迟 25-40%（Adaline Labs 生产数据，经第三方转述）。本项目自己的测量同向【实测，trace `router_eval_20261013_110645.json`】：LLM 二分类 direct/retrieve（自建 60 条标注集准确率 96.7%）；直答路线平均 14.4s / ¥0.0006 / 2 次调用 vs 检索路线 33.9s / ¥0.0037 / 3.2 次调用——快 2.4 倍、便宜 6 倍。

### 3.3 编排：LangGraph 成为默认

生产系统标准化使用 LangGraph 做 agentic loop 编排；checkpoint 让每步可追溯、可暂停（人工审批）、可复现【第三方】。本项目 pipeline.py 即 LangGraph 状态机【实测】。

### 3.4 MCP：检索的控制平面（2026 新共识）

- 语义分工：**Resources 承载只读检索，Tools 留给动作**（重建索引、触发同步、打第三方 API）【第三方实践文】。
- 生产路径：OpenAI Responses API 以 `mcp` tool 挂远程 MCP server（`allowed_tools` + `require_approval`）；Cloudflare 模式 = SQL 为 source of truth + Vectorize 存向量 + 结果重组回应用数据【第三方】。
- 安全模型随协议走：confused deputy（静态 client id）、token passthrough 禁止、OAuth metadata 发现打内网的 SSRF、共享队列 session hijacking【第三方】。
- 一句话定性：**MCP 不是检索算法，是让所有应用共享同一检索边界（ACL、元数据 join、权限）的协议**；2026 roadmap 重点是传输层无状态扩展与 `.well-known` 发现【第三方】。

### 3.5 上下文工程成为总名词

Anthropic 定义："the set of strategies for curating and maintaining the optimal set of tokens during LLM inference"（含 prompt 之外落到上下文里的一切：检索文档、历史、工具定义、长期记忆）【第三方转述官方】。RAG 是它的一个子系统；prompt 工程只管一句话，上下文工程管产生这句话的整条管线。四支柱：指令 / 检索 / 历史 / 工具定义【第三方】。

### 3.6 成本与延迟的实话（全部为第三方口径）

- token 成本 3-10× 简单 RAG；p95 延迟 1-2s → 4-15s（saaro）【第三方】
- 单厂商生产数据：客服解决率 +8-15%、内部 QA"我不知道" -10-25%、多跳质量 2×、总成本约 1.5×（callsphere）【第三方单厂商】
- 长上下文 vs RAG 的 2026 裁决：<约 200K token 的静态语料可全量塞（带缓存）；更大或常变的语料检索仍赢。成本数量级对比：RAG 约 $0.00008/query vs 全量上下文约 $0.10/query（byteiota 粗算口径）【第三方】。
- 采用面：57.3% 组织已有 agent 在生产；市场规模 $1.94B（2025）→ $9.86B（2030）（zylos 汇总）【第三方】。

---

## 4. 评测：基准、污染与协议纪律

### 4.1 污染是 2026 的显学

- GAIA 验证答案公开且年代久到可能已进训练数据；强 GAIA 分数应被怀疑【第三方】。
- search-time contamination：带搜索的 agent 在运行中能搜到基准自己的问题上下文/元数据/答案【第三方】。
- Stanford AI Index 2026【官方报告，经第三方镜像核阅】：广泛使用的 benchmark 构造低质（无效题目率从 MMLU Math 的 2% 到 GSM8K 的 42%）；Arena 排名部分反映对平台的适应而非通用能力；Llama 4 被疑用基准测试数据训练（Meta 否认）；前沿模型 top-4 在 Arena 上相差不到 25 Elo，竞争转向成本/可靠性/域性能。
- 对本项目：dev 调参 / test 只跑一次 A/B、bge 与基准时间线重叠主动披露——这套协议纪律属于**优于行业平均**的一侧，是面试叙事点。

### 4.2 RAG 评测成熟度 0-4 阶段【第三方框架】

Stage 0 无度量（"有时幻觉"但定位不到检索还是生成）→ 1 有 golden set → 2 组件级定位（context recall/precision + faithfulness，judge 对人校准）→ 3 进 CI 当门禁 → 4 生产采样监控 + golden set 持续扩充。起步运营目标：groundedness ≥90%、幻觉率 <5%（运营目标，非行业标准）。本项目的 judge 一致率评测（agreement 82.7% / precision 98.3%）【实测】就是 Stage 2 的 judge 校准动作。

### 4.3 人类偏好榜

RAG-QA Arena（win-ratio 制）：ArchRAG 在 writing/technology/science 等多域领先（sota2，2026-05）【第三方】；Contextual AI RAG Agent 71.2%（+5.4% vs baseline，zylos 转述）【第三方】。

---

## 5. 冷静面：什么时候不要 agentic

- **该用简单 RAG 的情形**：query 以单跳为主、语料小且维护好、答案是咨询性而非事务性、引用比行动重要、延迟成本预算紧。干净的知识库 + 混合检索 + 轻精排，经常打败没人监控的复杂 agent 图【第三方 a2afans】。
- **最狠的一句**："在弱检索器上叠 agent，通常是花更多钱自信地错"（a2afans 观点转述）【第三方】。
- **失败机制**（TowardsAI，标题党数字"90% 失败"不可信，但机制描述可信）：router 误路由、检索污染上下文、validator 放行幻觉、没有 observability 导致无法调试；LLM 调用从 2-3 次涨到 10-15 次【第三方】。
- **多跳才是瓶颈**：AgenticRAGTracer（2026-02）上 GPT-5 最难部分仅 22.6% EM；失败模式聚集为"过早收链"与"过度延伸"——检索基础设施必要但不充分，模型的多跳规划才是约束【第三方，未见原文】。
- **增益实证**：Microsoft AgenticRAG 实验在 WixQA（企业支持基准）上事实性 +13% 相对提升【第三方转述，未见原论文】。
- **企业决策树**（agentmarketcap 综合）【第三方】：单跳 + <1s + 准确率 >85% → 传统 RAG；幻觉来自近似检索 → 加检索验证（最高 ROI 升级）；多跳 + 引用核实 → agentic；富关系结构 → 图 + 向量混合；查询路由让三条路径并存按 cost-accuracy 分派。

---

## 6. 对 corrective-rag 的启示

### 6.1 我们在谱系里的位置

survey 七种架构占两席，且两席都已落地：**Corrective RAG**（grader + 重写回环，DuReader/RGB 公开基准实测）+ **Adaptive RAG**（`router.py` 二分类，自建标注集 96.7% + 直答/检索路线成本延迟实测）。

一个必要的张力说明：survey 把"分类器 + 流程"也归入 agentic 架构，而按"谁决定步骤序列"的严格判定线（00 号文档的诚实声明），本项目是 **workflow**——路由与纠错的每一步由代码和判断节点决定，不是模型自主规划。两种口径都成立，区别在于对外怎么说；面试被追问"这是不是 agent"时，主动给判定线比含混其辞得分。

### 6.2 与 2026 前沿的差距（诚实版）

| 维度 | 我们现状 | 2026 前沿 | 差距性质 |
|---|---|---|---|
| 检索决策 | prompt-based LLM judge | RL / proxy model 学习策略（SPARKLE、ZeroSearch） | 训练资源差距，非认知差距；我们的负面实证（rerank 分数饱和、FN 16.9%）正是前沿动机的实证 |
| 查询路由 | LLM 二分类 + 规则降级（96.7% 实测；直答快 2.4×/便宜 6×） | 同形态即 2026 共识（retrieve-or-skip）；前沿增量在多源路由（vector/SQL/web/KG 按 query 选） | 形态已对齐；差距在多源与 MCP 边界 |
| 多跳 | DuReader 主单跳；RGB integration 两要点 | AgenticRAGTracer 显示多跳规划是全行业瓶颈 | 同一起跑线，可自建小基准观察 |
| 记忆 | 滑窗 + 会话内摘要（第①②层） | Mem0/CMA/MAGMA 跨会话结构化记忆 | 明确下一步候选，非简历必需 |
| 评测协议 | dev/test 分离 + 污染披露 + judge 校准 | 行业普遍污染 + 口径混乱 | 我们优于平均 |
| 工具/多源 | 单一 FAISS 语料 | 混合检索 + SQL + web + MCP 多工具路由 | 工程扩展项 |

### 6.3 可落地下一步（按 ROI 排序）

1. **test split 一次终测（Phase 7）**：4,000 条只跑一次，协议纪律本身就是叙事；跑完简历 bullet 的 X/Y 全部到位。
2. **多跳自检**：借 AgenticRAGTracer 思路自建 20-30 条两跳问题，定位闭环在哪断（对应 RGB integration 73.0% 的弱点【实测】）。
3. **router 的下一步（可选）**：形态已对齐 2026 共识；若要再进一步，方向是多源路由（vector/SQL/web 按 query 分派），而非把二分类换成 RL——后者在单卡 8GB + 10 元余额的约束下是负分项。
4. **叙事**：四组公开基准数字（DuReader A/B / judge 一致率 / RGB 四场景 / router 96.7% + 成本延迟对比）+ 一个负面实证（reranker 分数不能当门控）+ 协议纪律，足以支撑"方法侧"简历项目；不需要追 RL。

### 6.4 面试可用的三句话

- "2026 的共识是把检索决策变成可学习的策略，但主流做法是把策略放在主模型外面（SPARKLE 的 proxy model）；我们在公开基准上实测了 prompt 式门控的天花板（judge 误杀 16.9%、reranker 分数饱和），也实测了轻量路由的收益（96.7%、快 2.4 倍/便宜 6 倍）。"
- "多跳推理是瓶颈不是检索（AgenticRAGTracer：GPT-5 最难多跳 22.6% EM），我们在 DuReader/RGB 上量化了自己系统的多跳弱点。"
- "评测协议比模型选择更重要：GAIA 污染、search-time contamination 是 2026 显学，我们 dev/test 分离 + 主动披露数据污染。"

---

## 7. 未验证清单（诚实边界）

- **本轮未打开原文、仅凭摘要或二手源**：BrowseComp 人类基线 29.2%（benchgen 转述 + 训练数据一致，论文正文未直核）；HCG-RAG 8-135×（中文二手）；Microsoft AgenticRAG WixQA +13%（width.ai 转述）；Tongyi DeepResearch 各基准分数（CSDN 转述技术报告）；Mem0 融资/ AWS 合作/ 80K 开发者（官方博客 + 第三方混合）；Continuum Memory Architecture 五原语与 MAGMA 四图（第三方转述）；AgenticRAGTracer 22.6% EM（第三方）；DRACO arXiv 编号（第三方）。
- **成本/采用率数字全部为厂商或聚合器口径，未独立复现**：3-10× 成本、4-15s p95、57.3% 采用率、$1.94B→$9.86B 市场、$6.27B→$28.45B 记忆市场、RAG $0.00008 vs 长上下文 $0.10/query。
- **口径冲突已识别的两处**：① 某第三方教程称 BrowseComp"人类专家 87.5%"，与论文人类基线 29.2% 矛盾，判定不可信未采用；② Kimi K2.5 BrowseComp 有 74.9%（官方 model card 口径）与 72.1%（第三方教程 harness）两个数，未能归一，本文采用官方口径。
- **来自训练数据、本轮未核验故未展开**：WebSailor / WebDancer / WebWalker（2025 开源深度研究训练线）、DeepResearcher / WebThinker（live search RL 训练）。
- **DeepResearch Bench II live leaderboard 64.38** 为 aievals.co 2026-10-01 快照，非官方 arXiv 数字。

## 8. 主要信源

- 官方 arXiv/ACL：2501.09136（Agentic RAG survey v4）、2603.07379（SoK, POMDP）、2503.09516（Search-R1）、2505.04588（ZeroSearch）、2506.02404（GraphRAG-Bench）、2504.12516（BrowseComp）、2508.06600（BrowseComp-Plus）、2601.08536（DeepResearch Bench II）、2504.19413（Mem0）、ACL 2026 SPARKLE（2026.acl-long.1793）、EACL 2026 ReflectiveRAG（2026.eacl-industry.27）、ACL 2025 KiRAG
- 官方机构：Stanford AI Index 2026（hai.stanford.edu）
- 第三方（实践/榜单/转述）：saaro.net、callsphere.ai、a2afans.com、width.ai、towardsai.net、agentmarketcap.ai、aievals.co、benchgen.com、sota2.com、zylos.ai、mem0.ai（官方博客）、llmwiki.hu、53ai.com、ima.qq.com 知识库文章
