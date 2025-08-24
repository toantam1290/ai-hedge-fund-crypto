from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning


def _pivot_points(df: pd.DataFrame, window: int = 3):
    highs = df["high"]
    lows = df["low"]
    pivoth = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    pivotl = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    return pivoth.dropna(), pivotl.dropna()


def _smc_signal(df: pd.DataFrame) -> Dict[str, Any]:
    pivoth, pivotl = _pivot_points(df, 3)
    if len(pivoth) < 2 or len(pivotl) < 2:
        return {"signal": "neutral", "confidence": 0.5, "metrics": {}}

    last_two_highs = pivoth.tail(2).values
    last_two_lows = pivotl.tail(2).values

    hh = last_two_highs[-1] > last_two_highs[-2]
    hl = last_two_lows[-1] > last_two_lows[-2]
    ll = last_two_lows[-1] < last_two_lows[-2]
    lh = last_two_highs[-1] < last_two_highs[-2]

    if hh and hl:
        return {"signal": "bullish", "confidence": 0.7, "metrics": {"hh": True, "hl": True}}
    if ll and lh:
        return {"signal": "bearish", "confidence": 0.7, "metrics": {"ll": True, "lh": True}}
    return {"signal": "neutral", "confidence": 0.5, "metrics": {}}


class SMCStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "SMCStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                if df.empty or len(df) < 20:
                    continue
                sig = _smc_signal(df)
                technical_analysis[ticker][interval.value] = {
                    "signal": sig["signal"],
                    "confidence": round(sig["confidence"] * 100),
                    "strategy_signals": {
                        "smc": {
                            "signal": sig["signal"],
                            "confidence": round(sig["confidence"] * 100),
                            "metrics": sig["metrics"],
                        }
                    },
                }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "SMC Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}


