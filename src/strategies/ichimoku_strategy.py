from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
from indicators import calculate_atr


def _ichimoku_signal(price_data: pd.DataFrame) -> Dict[str, Any]:
    high = price_data["high"]
    low = price_data["low"]
    close = price_data["close"]

    conversion_line = (high.rolling(9).max() + low.rolling(9).min()) / 2
    base_line = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((conversion_line + base_line) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

    atr = calculate_atr(price_data, 14)

    last_close = close.iloc[-1]
    last_tenkan = conversion_line.iloc[-1]
    last_kijun = base_line.iloc[-1]
    last_cloud_top = cloud_top.iloc[-1]
    last_cloud_bottom = cloud_bottom.iloc[-1]

    if pd.isna(last_cloud_top) or pd.isna(last_cloud_bottom) or pd.isna(last_tenkan) or pd.isna(last_kijun):
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    if last_close > last_cloud_top and last_tenkan > last_kijun:
        signal = "bullish"
    elif last_close < last_cloud_bottom and last_tenkan < last_kijun:
        signal = "bearish"
    else:
        signal = "neutral"

    cloud_thickness = abs(last_cloud_top - last_cloud_bottom)
    confidence = min(1.0, max(0.1, cloud_thickness / max(1e-8, atr.iloc[-1])))

    return {
        "signal": signal,
        "confidence": float(confidence),
        "metrics": {
            "tenkan": float(last_tenkan),
            "kijun": float(last_kijun),
            "cloud_top": float(last_cloud_top),
            "cloud_bottom": float(last_cloud_bottom),
        },
    }


class IchimokuStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "IchimokuStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                if df.empty or len(df) < 60:
                    continue
                ichimoku_result = _ichimoku_signal(df)
                technical_analysis[ticker][interval.value] = {
                    "signal": ichimoku_result["signal"],
                    "confidence": round(ichimoku_result["confidence"] * 100),
                    "strategy_signals": {
                        "ichimoku": {
                            "signal": ichimoku_result["signal"],
                            "confidence": round(ichimoku_result["confidence"] * 100),
                            "metrics": ichimoku_result["metrics"],
                        }
                    },
                }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "Ichimoku Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}


