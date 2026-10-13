"""纠错循环状态机（Phase 3 核心，整个项目的灵魂）。

图结构（Phase 6 起 router 在图头分流）：

    [history 非空] → condense → router ──direct──> generate(parametric) → END
    [history 为空]              │
                          （单轮直进）
                                  └──retrieve──> retrieve → rerank → judge ──sufficient──→ generate → END
                                                       ↑                                    │
                                                  （纠错回环）                              └─insufficient─→ iteration<MAX? ─yes→ rewrite → (回 retrieve)
                                                                                                    └no──→ abstain → END

router 节点（Phase 6 新增，Adaptive-RAG 的 classifier）：
- 二分类 direct/retrieve：闲聊/稳定常识直答（跳过检索/重排/judge），
  领域问题走完整纠错循环；解析失败一律 retrieve（安全侧，同规则路由偏向）
- 在 condense 之后：指代没消解连分类都没法做
- 论文三分类收敛为两分类的理由见 router.py docstring：多步由纠错循环承担，
  单趟 abstain 会放大 judge 16.9% 误杀率

condense 节点（Phase 5 新增，多轮对话的关键）：
- 只在 history 非空时触发，单轮问题直接透传——不为用不到的功能花钱；
- 把口语化+指代（"那它的成本呢"）改写成独立可检索的问题；
- 与 rewrite 节点的区别：condense 在入口、面向对话历史；rewrite 在
  重试时、面向检索失败。两者共用 LLMRewriter 和漂移自检。

为什么用状态机而不是 while 循环：
- 每个分支、每步 trace、终止条件都在图里显式可见，面试一张图讲完；
- while 循环的重试逻辑散落在代码里，加一个分支要通读全文还容易漏 trace；
- LangGraph 的 conditional edge + state counter 就是为"重试/回退"设计的。

关键设计（每条都能在面试里展开）：
1. rerank 在 judge 之前：先把相关性过滤掉，judge 只判"够不够答"，
   两层质检各司其职（CRAG 的 evaluator 之后的 RAG 评测核心设计）。
2. abstain 是一等终态，不是异常分支：宁可不答，不牵强。
   MAX_ITERATIONS 次还不行就诚实说不知道，并附上尝试记录。
3. judge 和 rewriter 都可插拔（LLM / 规则），无 API key 全链路照跑。
4. trace 逐步落 state：延迟、成本、每步评分，简历数字的唯一出处。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from corrective_rag import config  # noqa: E402


class RAGState(TypedDict, total=False):
    question: str
    query: str
    history: str  # 渲染好的对话历史（Phase 5：多轮上下文）
    route: str  # 路由决策：direct / retrieve（Phase 6）
    retrieved: List[Dict]
    ranked: List[Dict]
    judge: Dict
    answer: str
    status: str
    iteration: int
    trace: List[Dict]


def _log(state: RAGState, step: str, **fields) -> None:
    entry = {"step": step, "t": round(time.time() - state.get("_t0", time.time()), 3)}
    entry.update(fields)
    state.setdefault("trace", []).append(entry)


class CorrectiveRAG:
    """纠错式 RAG 流水线（LangGraph 编译后的可调用对象）。"""

    def __init__(
        self,
        retriever=None,
        reranker=None,
        judge=None,
        rewriter=None,
        generator=None,
        router=None,
        judge_threshold: float = 0.0,
    ) -> None:
        from langgraph.graph import END, START, StateGraph

        from corrective_rag.generate import generate as gen_fn
        from corrective_rag.judge import RuleBasedJudge, get_judge
        from corrective_rag.rerank import get_reranker
        from corrective_rag.retrieve import Retriever
        from corrective_rag.rewrite import get_rewriter
        from corrective_rag.router import get_router

        self.retriever = retriever or Retriever()
        self.reranker = reranker or get_reranker()
        self.judge = judge or get_judge(threshold=judge_threshold)
        self.rewriter = rewriter or get_rewriter()
        self.router = router or get_router()
        self.generate = generator or gen_fn

        # ---- 节点定义 ----
        def node_condense(state: RAGState) -> RAGState:
            """多轮入口：history 非空时把本轮问题 condense 成独立问题。

            单轮（history 空）直接透传——不花这个调用的钱。
            改写失败/漂移过度时保留原问题（_checked 已保证）。
            """
            history = state.get("history", "")
            if not history:
                return {"trace": state["trace"]}
            t0 = time.time()
            result = self.rewriter.rewrite(
                state["question"], retrieved=None, history=history
            )
            _log(
                state,
                "condense",
                applied=result.get("applied"),
                strategy=result.get("strategy"),
                reason=result.get("reason"),
                new_query=result.get("rewritten"),
                latency=round(time.time() - t0, 3),
            )
            return {"query": result["rewritten"], "trace": state["trace"]}

        def node_router(state: RAGState) -> RAGState:
            """查询路由（Phase 6）：direct 直答 / retrieve 进纠错循环。

            判 direct 的条件很窄（闲聊/问候/稳定常识），解析失败和安全
            默认都落 retrieve——误判 direct 的领域问题会得到无依据答案，
            误判 retrieve 只是多花一次检索，代价不对称。
            """
            t0 = time.time()
            decision = self.router.route(state["query"])
            _log(
                state,
                "router",
                route=decision.get("route"),
                reason=decision.get("reason"),
                classifier=decision.get("classifier"),
                latency=round(time.time() - t0, 3),
            )
            return {"route": decision["route"], "trace": state["trace"]}

        def node_retrieve(state: RAGState) -> RAGState:
            query = state["query"]
            t0 = time.time()
            retrieved = self.retriever.retrieve(
                query, top_k=config.TOP_K_RETRIEVE
            )
            _log(
                state,
                "retrieve",
                query=query,
                n=len(retrieved),
                top1_score=retrieved[0]["score"] if retrieved else None,
                latency=round(time.time() - t0, 3),
            )
            return {"retrieved": retrieved, "trace": state["trace"]}

        def node_rerank(state: RAGState) -> RAGState:
            t0 = time.time()
            ranked = self.reranker.rerank(
                state["query"], state["retrieved"], top_k=config.TOP_K_FINAL
            )
            _log(
                state,
                "rerank",
                n=len(ranked),
                scores=[r.get("rerank_score") for r in ranked],
                latency=round(time.time() - t0, 3),
            )
            return {"ranked": ranked, "trace": state["trace"]}

        def node_judge(state: RAGState) -> RAGState:
            t0 = time.time()
            verdict = self.judge.judge(state["question"], state["ranked"])
            _log(
                state,
                "judge",
                level=verdict.get("level"),
                sufficient=verdict.get("sufficient"),
                reason=verdict.get("reason"),
                usage=verdict.get("usage"),
                latency=round(time.time() - t0, 3),
            )
            return {"judge": verdict, "trace": state["trace"]}

        def node_rewrite(state: RAGState) -> RAGState:
            it = state.get("iteration", 0) + 1
            result = self.rewriter.rewrite(
                state["question"],
                state.get("ranked") or state.get("retrieved"),
                history=state.get("history", ""),
            )
            _log(
                state,
                "rewrite",
                iteration=it,
                applied=result.get("applied"),
                strategy=result.get("strategy"),
                reason=result.get("reason"),
                new_query=result.get("rewritten"),
            )
            return {
                "query": result["rewritten"],
                "iteration": it,
                "trace": state["trace"],
            }

        def node_generate(state: RAGState) -> RAGState:
            from corrective_rag.cost import tracker

            t0 = time.time()
            # resolved_query = condense/rewrite 消解指代后的查询。
            # 传给生成器，避免\"系统刚消解了'它'，答案又装不知道\"的笑话
            # （Phase 5 demo 实测教训：condense 消解出实体后，生成器仍收
            # 原始问题，会闹出系统刚消解又装不知道的笑话。judge 仍用原始
            # question 判断：用户的原问题能否被资料回答，才是门控该问的）
            parametric = state.get("route") == "direct"
            answer = self.generate(
                state["question"],
                state["ranked"],
                resolved_query=state.get("query", ""),
                parametric=parametric,
            )
            _log(
                state,
                "generate",
                latency=round(time.time() - t0, 3),
                cost_so_far=tracker.snapshot(),
            )
            return {"answer": answer, "status": "answered", "trace": state["trace"]}

        def node_abstain(state: RAGState) -> RAGState:
            judge_info = state.get("judge", {})
            answer = (
                "根据现有资料无法可靠回答这个问题。"
                f"（已重试 {state.get('iteration', 0)} 次，"
                f"最后判定：{judge_info.get('reason', '资料不足')}）"
            )
            _log(state, "abstain", reason=judge_info.get("reason"))
            return {"answer": answer, "status": "abstained", "trace": state["trace"]}

        # ---- 条件分支 ----
        def route_after_router(state: RAGState) -> str:
            return "generate" if state.get("route") == "direct" else "retrieve"

        def route_after_judge(state: RAGState) -> str:
            if state["judge"].get("sufficient"):
                return "generate"
            if state.get("iteration", 0) >= config.MAX_ITERATIONS:
                return "abstain"
            return "rewrite"

        graph = StateGraph(RAGState)
        graph.add_node("condense", node_condense)
        graph.add_node("router", node_router)
        graph.add_node("retrieve", node_retrieve)
        graph.add_node("rerank", node_rerank)
        graph.add_node("judge", node_judge)
        graph.add_node("rewrite", node_rewrite)
        graph.add_node("generate", node_generate)
        graph.add_node("abstain", node_abstain)

        graph.add_edge(START, "condense")
        graph.add_edge("condense", "router")
        graph.add_conditional_edges(
            "router",
            route_after_router,
            {"generate": "generate", "retrieve": "retrieve"},
        )
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "judge")
        graph.add_conditional_edges(
            "judge",
            route_after_judge,
            {"generate": "generate", "rewrite": "rewrite", "abstain": "abstain"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)
        graph.add_edge("abstain", END)

        self.graph = graph.compile()

    def run(self, question: str, history: str = "") -> RAGState:
        """history：渲染好的对话历史文本（由 ConversationMemory.get_history()
        产出）。空字符串 = 单轮，condense 节点直接透传不花钱。"""
        init: RAGState = {
            "question": question,
            "query": question,
            "history": history,
            "route": "",
            "retrieved": [],
            "ranked": [],
            "judge": {},
            "answer": "",
            "status": "unknown",
            "iteration": 0,
            "trace": [],
            "_t0": time.time(),
        }
        return self.graph.invoke(init)

    def draw(self) -> str:
        """导出图的 mermaid 文本（README / 面试画图用）。"""
        return self.graph.get_graph().draw_mermaid()
