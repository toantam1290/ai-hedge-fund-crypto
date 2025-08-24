from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
from indicators import calculate_atr


def _supertrend_signal(price_data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Dict[str, Any]:
    atr = calculate_atr(price_data, period)
    hl2 = (price_data["high"] + price_data["low"]) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    close = price_data["close"]
    last_close = close.iloc[-1]
    last_upper = upper_basic.iloc[-1]
    last_lower = lower_basic.iloc[-1]

    if pd.isna(last_upper) or pd.isna(last_lower):
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    if last_close > last_lower:
        signal = "bullish"
        confidence = min(1.0, max(0.1, (last_close - last_lower) / max(1e-8, atr.iloc[-1])))
    elif last_close < last_upper:
        signal = "bearish"
        confidence = min(1.0, max(0.1, (last_upper - last_close) / max(1e-8, atr.iloc[-1])))
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": float(confidence),
        "metrics": {
            "atr": float(atr.iloc[-1]),
            "upper_basic": float(last_upper),
            "lower_basic": float(last_lower),
        },
    }


class SuperTrendStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "SuperTrendStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                if df.empty or len(df) < 50:
                    continue
                st = _supertrend_signal(df, period=10, multiplier=3.0)
                technical_analysis[ticker][interval.value] = {
                    "signal": st["signal"],
                    "confidence": round(st["confidence"] * 100),
                    "strategy_signals": {
                        "supertrend": {
                            "signal": st["signal"],
                            "confidence": round(st["confidence"] * 100),
                            "metrics": st["metrics"],
                        }
                    },
                }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "SuperTrend Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}


