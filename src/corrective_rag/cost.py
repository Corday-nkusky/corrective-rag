"""API 成本核算：所有 LLM 调用的 token 用量与费用汇总。

为什么单独一个模块：
- "每 query 平均成本 ¥X" 是简历/面试必问数字，必须从 trace 里出，不能估；
- 按量付费账号（余额有限）需要能看着花钱，超支前能停；
- 价格是供应商属性，集中一处，换模型只改价目表。

价格来源：小米 MiMo 开放平台官方定价页
（platform.xiaomimimo.com/docs/zh-CN/pricing，2026-10-13 查证）：
    mimo-v2.6-flash 国内：输入(未命中缓存) ¥1.00 / 百万 tokens，
                          输入(命中缓存) ¥0.02，输出 ¥2.00。
缓存命中价暂不细分（judge/generate 的 prompt 前缀不稳定，命中率低，
按未命中价计是保守上界——宁可多算，不可少算）。

env 覆盖：LLM_PRICE_IN / LLM_PRICE_OUT（元/百万 tokens）。
"""

from __future__ import annotations

import os
import threading

# 价目表（元 / 百万 tokens）：模型名 -> (输入, 输出)
PRICE_TABLE = {
    "mimo-v2.6-flash": (1.00, 2.00),
    "mimo-v2.6-pro": (3.00, 6.00),
}

DEFAULT_PRICE = (1.00, 2.00)  # 未知模型按 flash 价保守估


def _price_for(model: str):
    return PRICE_TABLE.get(model, DEFAULT_PRICE)


class CostTracker:
    """线程安全的累计计数器。record(usage) 由各 LLM 调用点调用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def record(self, usage, model: str = "") -> None:
        if not usage:
            return
        p = getattr(usage, "prompt_tokens", None) or (
            usage.get("prompt_tokens") if isinstance(usage, dict) else None
        )
        c = getattr(usage, "completion_tokens", None) or (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        if p is None and c is None:
            return
        with self._lock:
            self.prompt_tokens += int(p or 0)
            self.completion_tokens += int(c or 0)
            self.calls += 1

    def cost_cny(self, model: str = "") -> float:
        price_in, price_out = _price_for(model)
        return (
            self.prompt_tokens * price_in + self.completion_tokens * price_out
        ) / 1_000_000

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_cny": round(self.cost_cny(), 4),
        }

    def reset(self) -> None:
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.calls = 0


# 进程级单例：judge / generate 各调用点共享一份账
tracker = CostTracker()


def record(usage, model: str = "") -> None:
    tracker.record(usage, model)


def price_in(model: str) -> float:
    return float(
        os.getenv("LLM_PRICE_IN", _price_for(model)[0])
    )


def price_out(model: str) -> float:
    return float(
        os.getenv("LLM_PRICE_OUT", _price_for(model)[1])
    )
