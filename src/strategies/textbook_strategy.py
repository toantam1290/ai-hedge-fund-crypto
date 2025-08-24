from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
from indicators import calculate_ema, calculate_rsi


def _textbook_signal(df: pd.DataFrame) -> Dict[str, Any]:
    ema50 = calculate_ema(df, 50)
    ema200 = calculate_ema(df, 200)
    rsi14 = calculate_rsi(df, 14)
    close = df["close"]

    last_close = close.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    last_ema200 = ema200.iloc[-1]
    last_rsi = rsi14.iloc[-1]

    if pd.isna(last_ema50) or pd.isna(last_ema200) or pd.isna(last_rsi):
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    if last_ema50 > last_ema200 and last_rsi > 50 and last_close > last_ema50:
        return {
            "signal": "bullish",
            "confidence": 0.8,
            "metrics": {"ema50": float(last_ema50), "ema200": float(last_ema200), "rsi14": float(last_rsi)},
        }
    if last_ema50 < last_ema200 and last_rsi < 50 and last_close < last_ema50:
        return {
            "signal": "bearish",
            "confidence": 0.8,
            "metrics": {"ema50": float(last_ema50), "ema200": float(last_ema200), "rsi14": float(last_rsi)},
        }

    return {
        "signal": "neutral",
        "confidence": 0.5,
        "metrics": {"ema50": float(last_ema50), "ema200": float(last_ema200), "rsi14": float(last_rsi)},
    }


class TextbookStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "TextbookStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                if df.empty or len(df) < 220:
                    continue
                sig = _textbook_signal(df)
                technical_analysis[ticker][interval.value] = {
                    "signal": sig["signal"],
                    "confidence": round(sig["confidence"] * 100),
                    "strategy_signals": {
                        "textbook": {
                            "signal": sig["signal"],
                            "confidence": round(sig["confidence"] * 100),
                            "metrics": sig["metrics"],
                        }
                    },
                }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "Textbook Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}


