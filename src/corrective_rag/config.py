"""集中配置：路径、模型名、API 参数、切分与检索超参。

原则：切模型 / 调 chunk 大小 / 换 embedding 维度只改这一个文件和 .env，
代码里不出现第二份硬编码。Phase 0 即可运行（python-dotenv 未安装时自动跳过）。
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 路径（ROOT = 仓库根目录，即 D:/code/corrective-rag）
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = ROOT / "data" / "corpus"
EVAL_DIR = ROOT / "data" / "eval"
INDEX_DIR = ROOT / "data" / "index"
TRACE_DIR = ROOT / "data" / "traces"


def _load_dotenv() -> None:
    """加载 .env；python-dotenv 未安装时静默跳过（smoke 阶段不依赖它）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


_load_dotenv()

# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
# HuggingFace 镜像：本机直连 huggingface.co 不通（2026-10-12 实测超时），
# 必须在 import sentence_transformers 之前设置 HF_ENDPOINT。
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

# 强制离线加载本地缓存的模型。sentence-transformers 每次构造都会联网向 HF Hub
# 查一次版本（即使权重早已缓存在 ~/.cache/huggingface），这笔网络往返极贵：
# bge-small-zh 加载 21.07s → 1.09s、bge-reranker-base 10.47s → 3.41s
# （2026-10-14 本机实测），冷启动合计 ~36s → ~8s。
# 需要下载新模型时，在 .env 里设 HF_HUB_OFFLINE=0 临时关闭。
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", HF_HUB_OFFLINE)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# bge 系列的官方约定：query 侧要加指令前缀，document 侧不加。
# 漏加这个前缀检索效果会明显掉（官方 README）。
BGE_QUERY_INSTRUCTION = os.getenv(
    "BGE_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
)

# --------------------------------------------------------------------------
# LLM（OpenAI 兼容协议；当前用小米 MiMo，换供应商只改 .env 不动代码）
# --------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.xiaomimimo.com/v1")
# DEEPSEEK_API_KEY 是兼容回落：老 .env 只写它也能跑，新配置一律写 LLM_API_KEY
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mimo-v2.6-flash")

# 深度思考（思维链）开关。对 mimo-v2.6-flash / pro / v2.5 系列，官方文档写明
# **默认开启**，而它有两个副作用：
#   ① 生成前先跑思维链，延迟上升（实测等首字 1.47s → 2.41s）；
#   ② temperature / top_p 被强制成 1.0 / 0.95，传了也不生效——对 judge /
#      router 这类要求确定性的判定调用是硬伤。
# 取值：disabled（关，默认）｜enabled（开）｜空字符串（不发送该参数，
# 换到不支持 thinking 字段的供应商时用）。
LLM_THINKING = os.getenv("LLM_THINKING", "disabled")


def llm_extra_body() -> dict:
    """深度思考开关。thinking 不是标准 OpenAI 参数，必须走 extra_body
    （官方文档：通过 OpenAI SDK 传 thinking 必须放进 extra_body）。"""
    if not LLM_THINKING:
        return {}
    return {"thinking": {"type": LLM_THINKING}}

# --------------------------------------------------------------------------
# 切分与检索超参
# --------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))  # 先多取，交给 reranker 精选
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "4"))         # 最终进 prompt 的文档数
JUDGE_RELEVANCE_THRESHOLD = int(os.getenv("JUDGE_RELEVANCE_THRESHOLD", "1"))  # 0/1/2 口径
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))   # 纠错循环上限，防成本爆炸
