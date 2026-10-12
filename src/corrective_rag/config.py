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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# bge 系列的官方约定：query 侧要加指令前缀，document 侧不加。
# 漏加这个前缀检索效果会明显掉（官方 README）。
BGE_QUERY_INSTRUCTION = os.getenv(
    "BGE_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
)

# --------------------------------------------------------------------------
# LLM（DeepSeek，OpenAI 兼容协议）
# --------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# --------------------------------------------------------------------------
# 切分与检索超参
# --------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))

TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", "20"))  # 先多取，交给 reranker 精选
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "4"))         # 最终进 prompt 的文档数
JUDGE_RELEVANCE_THRESHOLD = int(os.getenv("JUDGE_RELEVANCE_THRESHOLD", "1"))  # 0/1/2 口径
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))   # 纠错循环上限，防成本爆炸
