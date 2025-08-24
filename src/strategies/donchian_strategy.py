from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning


def _donchian_signal(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    upper = df["high"].rolling(window).max()
    lower = df["low"].rolling(window).min()
    mid = (upper + lower) / 2
    close = df["close"]

    last_close = close.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    last_mid = mid.iloc[-1]

    if pd.isna(last_upper) or pd.isna(last_lower):
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    if last_close > last_upper:
        signal = "bullish"
        confidence = 0.8
    elif last_close < last_lower:
        signal = "bearish"
        confidence = 0.8
    else:
        signal = "neutral"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "upper": float(last_upper),
            "lower": float(last_lower),
            "mid": float(last_mid),
        },
    }


class DonchianStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "DonchianStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                if df.empty or len(df) < 25:
                    continue
                sig = _donchian_signal(df, window=20)
                technical_analysis[ticker][interval.value] = {
                    "signal": sig["signal"],
                    "confidence": round(sig["confidence"] * 100),
                    "strategy_signals": {
                        "donchian": {
                            "signal": sig["signal"],
                            "confidence": round(sig["confidence"] * 100),
                            "metrics": sig["metrics"],
                        }
                    },
                }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "Donchian Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}


