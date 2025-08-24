from typing import Dict, Any
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
from indicators import calculate_rsi


class RSIStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "RSIStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                if df.empty or len(df) < 30:
                    continue

                rsi14 = calculate_rsi(df, 14)
                rsi28 = calculate_rsi(df, 28)

                last_rsi14 = float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else 50.0
                last_rsi28 = float(rsi28.iloc[-1]) if not pd.isna(rsi28.iloc[-1]) else 50.0

                # Simple RSI regime rules
                if last_rsi14 >= 60 and last_rsi14 >= last_rsi28:
                    signal = "bullish"
                elif last_rsi14 <= 40 and last_rsi14 <= last_rsi28:
                    signal = "bearish"
                else:
                    signal = "neutral"

                # Confidence based on distance from 50 and agreement between RSI14 and RSI28
                dist = abs(last_rsi14 - 50.0) / 50.0  # 0..1
                agreement = 1.0 if (signal != "neutral" and ((signal == "bullish" and last_rsi28 >= 50) or (signal == "bearish" and last_rsi28 <= 50))) else 0.5
                confidence = max(0.1, min(1.0, 0.6 * dist + 0.4 * agreement))

                technical_analysis[ticker][interval.value] = {
                    "signal": signal,
                    "confidence": round(confidence * 100),
                    "strategy_signals": {
                        "rsi": {
                            "signal": signal,
                            "confidence": round(confidence * 100),
                            "metrics": {
                                "rsi14": last_rsi14,
                                "rsi28": last_rsi28,
                                "overbought": last_rsi14 >= 70,
                                "oversold": last_rsi14 <= 30,
                            },
                        }
                    },
                }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "RSI Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}
