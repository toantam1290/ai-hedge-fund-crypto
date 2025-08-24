from typing import Dict, Any, List, Tuple
import json
import pandas as pd
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
from indicators import calculate_ema, calculate_rsi, calculate_atr, calculate_adx


TF_ORDER = {"5m": 1, "15m": 2, "30m": 3, "1h": 4, "4h": 5, "1d": 6, "3d": 7, "1w": 8}


def _pick_higher_lower(intervals: List) -> Tuple[str, str]:
    values = [iv.value for iv in intervals]
    values = [v for v in values if v in TF_ORDER]
    if not values:
        return None, None
    values_sorted = sorted(values, key=lambda x: TF_ORDER[x])
    lower = values_sorted[0]
    higher = values_sorted[-1]
    return higher, lower


def _ichimoku_bias(df: pd.DataFrame):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

    last_close = close.iloc[-1]
    last_tenkan = tenkan.iloc[-1]
    last_kijun = kijun.iloc[-1]
    last_top = cloud_top.iloc[-1]
    last_bot = cloud_bottom.iloc[-1]

    if pd.isna(last_top) or pd.isna(last_bot) or pd.isna(last_tenkan) or pd.isna(last_kijun):
        return "neutral", 0.5
    if last_close > last_top and last_tenkan > last_kijun:
        return "bullish", 0.7
    if last_close < last_bot and last_tenkan < last_kijun:
        return "bearish", 0.7
    return "neutral", 0.5


def _supertrend_dir(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    atr = calculate_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    last_close = df["close"].iloc[-1]
    last_upper = upper_basic.iloc[-1]
    last_lower = lower_basic.iloc[-1]
    if pd.isna(last_upper) or pd.isna(last_lower):
        return "neutral", 0.5
    if last_close > last_lower:
        return "bullish", min(1.0, max(0.4, (last_close - last_lower) / max(1e-8, atr.iloc[-1]) * 0.3 + 0.5))
    if last_close < last_upper:
        return "bearish", min(1.0, max(0.4, (last_upper - last_close) / max(1e-8, atr.iloc[-1]) * 0.3 + 0.5))
    return "neutral", 0.5


def _donchian_ctx(df: pd.DataFrame, window: int = 20):
    upper = df["high"].rolling(window).max()
    lower = df["low"].rolling(window).min()
    mid = (upper + lower) / 2
    return upper, lower, mid


def _trend_on_higher(df: pd.DataFrame):
    ema50 = calculate_ema(df, 50)
    ema200 = calculate_ema(df, 200)
    adx_df = calculate_adx(df, 14)

    ema_bias = "neutral"
    if len(ema200.dropna()) > 0:
        if ema50.iloc[-1] > ema200.iloc[-1]:
            ema_bias = "bullish"
        elif ema50.iloc[-1] < ema200.iloc[-1]:
            ema_bias = "bearish"

    ich_bias, ich_conf = _ichimoku_bias(df)
    st_dir, st_conf = _supertrend_dir(df)
    upper, lower, mid = _donchian_ctx(df, 20)

    pos = 0.0
    last_close = df["close"].iloc[-1]
    if not pd.isna(upper.iloc[-1]) and not pd.isna(lower.iloc[-1]):
        span = max(1e-8, (upper.iloc[-1] - lower.iloc[-1]))
        pos = (last_close - lower.iloc[-1]) / span  # 0..1 within channel

    # voting
    score = 0.0
    def vote(val, weight):
        nonlocal score
        if val == "bullish":
            score += 1.0 * weight
        elif val == "bearish":
            score -= 1.0 * weight

    vote(ema_bias, 0.35)
    vote(ich_bias, 0.25)
    vote(st_dir, 0.25)
    # Donchian position: >0.6 bullish, <0.4 bearish
    if pos > 0.6:
        score += 0.15
    elif pos < 0.4:
        score -= 0.15

    # ADX strength scaling
    adx_val = float(adx_df["adx"].iloc[-1]) if "adx" in adx_df.columns else 20.0
    strength = min(1.0, max(0.4, adx_val / 50.0))
    final = score * strength

    if final > 0.15:
        return "bullish", min(1.0, 0.5 + final)
    if final < -0.15:
        return "bearish", min(1.0, 0.5 + abs(final))
    return "neutral", 0.5


def _entry_on_lower(df: pd.DataFrame, bias: str):
    ema20 = calculate_ema(df, 20)
    rsi14 = calculate_rsi(df, 14)
    atr14 = calculate_atr(df, 14)
    close = df["close"]
    upper, lower, mid = _donchian_ctx(df, 20)
    vol = df.get("volume")
    vol_ma20 = vol.rolling(20).mean() if vol is not None else None

    if len(ema20.dropna()) == 0 or len(rsi14.dropna()) == 0 or len(atr14.dropna()) == 0:
        return {"entry_signal": "neutral"}

    last_close = close.iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_rsi = rsi14.iloc[-1]
    last_atr = atr14.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    vol_spike = None
    if vol_ma20 is not None and not pd.isna(vol_ma20.iloc[-1]) and vol.iloc[-1] is not None:
        vol_spike = float(vol.iloc[-1] / max(1e-8, vol_ma20.iloc[-1]))

    # Breakout logic (Donchian + volume)
    if bias == "bullish" and not pd.isna(last_upper):
        breakout = last_close > last_upper and (vol_spike is None or vol_spike >= 1.3)
        if breakout:
            sl = float(last_close - 1.5 * last_atr)
            tp = float(last_close + 2.5 * last_atr)
            return {"entry_signal": "bullish", "entry_type": "breakout", "sl": sl, "tp": tp, "atr": float(last_atr), "vol_spike": vol_spike}

    if bias == "bearish" and not pd.isna(last_lower):
        breakdown = last_close < last_lower and (vol_spike is None or vol_spike >= 1.3)
        if breakdown:
            sl = float(last_close + 1.5 * last_atr)
            tp = float(last_close - 2.5 * last_atr)
            return {"entry_signal": "bearish", "entry_type": "breakdown", "sl": sl, "tp": tp, "atr": float(last_atr), "vol_spike": vol_spike}

    # Pullback logic (EMA20 + RSI + momentum)
    if bias == "bullish":
        cond = (last_close > last_ema20) and (last_rsi >= 50) and (close.iloc[-1] > close.iloc[-2])
        if cond:
            sl = float(last_close - 1.3 * last_atr)
            tp = float(last_close + 2.0 * last_atr)
            return {"entry_signal": "bullish", "entry_type": "pullback", "sl": sl, "tp": tp, "atr": float(last_atr), "vol_spike": vol_spike}

    if bias == "bearish":
        cond = (last_close < last_ema20) and (last_rsi <= 50) and (close.iloc[-1] < close.iloc[-2])
        if cond:
            sl = float(last_close + 1.3 * last_atr)
            tp = float(last_close - 2.0 * last_atr)
            return {"entry_signal": "bearish", "entry_type": "pullback", "sl": sl, "tp": tp, "atr": float(last_atr), "vol_spike": vol_spike}

    return {"entry_signal": "neutral"}


class MTFTrendEntryStrategy(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        data = state.get("data", {})
        data["name"] = "MTFTrendEntryStrategy"

        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        higher_tf, lower_tf = _pick_higher_lower(intervals)
        technical_analysis: Dict[str, Dict[str, Any]] = {t: {} for t in tickers}

        for ticker in tickers:
            df_h = data.get(f"{ticker}_{higher_tf}", pd.DataFrame()) if higher_tf else pd.DataFrame()
            df_l = data.get(f"{ticker}_{lower_tf}", pd.DataFrame()) if lower_tf else pd.DataFrame()

            if df_h.empty or df_l.empty:
                df = df_l if not df_l.empty else data.get(f"{ticker}_{intervals[0].value}", pd.DataFrame())
                if df.empty:
                    continue
                signal = "neutral"
                conf = 50
                entry = {"entry_signal": "neutral"}
                technical_analysis[ticker][lower_tf or intervals[0].value] = {
                    "signal": signal,
                    "confidence": conf,
                    "strategy_signals": {
                        "mtf_trend_entry": {"signal": signal, "confidence": conf, "metrics": entry}
                    },
                }
                continue

            bias, bias_conf = _trend_on_higher(df_h)
            entry = _entry_on_lower(df_l, bias)

            if entry["entry_signal"] == "bullish":
                signal = "bullish"
                conf = int(bias_conf * 100)
            elif entry["entry_signal"] == "bearish":
                signal = "bearish"
                conf = int(bias_conf * 100)
            else:
                signal = "neutral"
                conf = 50

            technical_analysis[ticker][lower_tf] = {
                "signal": signal,
                "confidence": conf,
                "strategy_signals": {
                    "mtf_trend_entry": {
                        "signal": entry["entry_signal"],
                        "confidence": conf,
                        "metrics": {
                            "bias": bias,
                            "sl": entry.get("sl"),
                            "tp": entry.get("tp"),
                            "atr": entry.get("atr"),
                            "entry_type": entry.get("entry_type"),
                            "vol_spike": entry.get("vol_spike"),
                            "higher_tf": higher_tf,
                            "lower_tf": lower_tf,
                        },
                    }
                },
            }

        message = HumanMessage(content=json.dumps(technical_analysis), name="technical_analyst_agent")
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "MTF Trend+Entry Strategy")
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis
        return {"messages": [message], "data": data}


