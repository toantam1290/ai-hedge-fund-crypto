from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
from indicators import calculate_ema


def _volume_signal(df: pd.DataFrame) -> Dict[str, Any]:
    vol = df["volume"]
    vol_ma = vol.rolling(20).mean()
    ema20 = calculate_ema(df, 20)
    close = df["close"]

    last_vol = vol.iloc[-1]
    last_vol_ma = vol_ma.iloc[-1]
    last_close = close.iloc[-1]
    last_ema20 = ema20.iloc[-1]

    if pd.isna(last_vol_ma) or pd.isna(last_ema20):
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    spike_ratio = float(last_vol / max(1e-8, last_vol_ma))

    if spike_ratio > 1.5 and last_close > last_ema20:
        return {"signal": "bullish", "confidence": 0.7, "metrics": {"vol_spike": spike_ratio}}
    if spike_ratio > 1.5 and last_close < last_ema20:
        return {"signal": "bearish", "confidence": 0.7, "metrics": {"vol_spike": spike_ratio}}

    return {"signal": "neutral", "confidence": 0.5, "metrics": {"vol_spike": spike_ratio}}


class VolumeStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "VolumeStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                if df.empty or len(df) < 30:
                    continue
                sig = _volume_signal(df)
                technical_analysis[ticker][interval.value] = {
                    "signal": sig["signal"],
                    "confidence": round(sig["confidence"] * 100),
                    "strategy_signals": {
                        "volume": {
                            "signal": sig["signal"],
                            "confidence": round(sig["confidence"] * 100),
                            "metrics": sig["metrics"],
                        }
                    },
                }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "Volume Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}


