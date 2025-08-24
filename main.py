import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dotenv import load_dotenv
from src.utils import settings
from datetime import datetime
from src.agent import Agent
from src.backtest.backtester import Backtester
from src.utils.telegram_notifier import TelegramNotifier


load_dotenv()

if __name__ == "__main__":

    if settings.mode == "backtest":
        backtester = Backtester(
            primary_interval=settings.primary_interval,
            intervals=settings.signals.intervals,
            tickers=settings.signals.tickers,
            start_date=settings.start_date,
            end_date=settings.end_date,
            initial_capital=settings.initial_cash,
            strategies=settings.signals.strategies,
            show_agent_graph=settings.show_agent_graph,
            show_reasoning=settings.show_reasoning,
            model_name=settings.model.name,
            model_provider=settings.model.provider,
            model_base_url=settings.model.base_url,
        )
        print("Starting backtest...")
        performance_metrics = backtester.run_backtest()
        performance_df = backtester.analyze_performance()

    else:
        # Reuse the Backtester logic for portfolio, trade execution, and PnL
        live_bt = Backtester(
            primary_interval=settings.primary_interval,
            intervals=settings.signals.intervals,
            tickers=settings.signals.tickers,
            start_date=settings.start_date,
            end_date=settings.end_date,
            initial_capital=settings.initial_cash,
            strategies=settings.signals.strategies,
            show_agent_graph=settings.show_agent_graph,
            show_reasoning=settings.show_reasoning,
            model_name=settings.model.name,
            model_provider=settings.model.provider,
            model_base_url=settings.model.base_url,
            initial_margin_requirement=settings.margin_requirement,
        )

        portfolio = live_bt.portfolio
        poll = int(settings.live_poll_seconds or 0)
        notifier = TelegramNotifier()

        import time
        while True:
            now_ts = datetime.now()
            result = Agent.run(
                primary_interval=settings.primary_interval,
                intervals=settings.signals.intervals,
                tickers=settings.signals.tickers,
                end_date=now_ts,
                portfolio=portfolio,
                strategies=settings.signals.strategies,
                show_reasoning=settings.show_reasoning,
                show_agent_graph=settings.show_agent_graph,
                model_name=settings.model.name,
                model_provider=settings.model.provider,
                model_base_url=settings.model.base_url,
            )

            decisions = result.get("decisions", {}) or {}
            analyst_signals = result.get("analyst_signals", {}) or {}
            risk_signals = analyst_signals.get("risk_management_agent", {}) or {}
            # Use current price per ticker from risk node; fallback to 0.0
            current_prices = {
                t: float(risk_signals.get(t, {}).get("current_price", 0.0)) for t in settings.signals.tickers
            }

            # Execute trades like in backtest
            executed_trades = {}
            for t in settings.signals.tickers:
                d = decisions.get(t, {}) or {}
                action = d.get("action", "hold")
                quantity = float(d.get("quantity", 0.0) or 0.0)
                price = current_prices.get(t, 0.0)
                executed_qty = live_bt.execute_trade(t, action, quantity, price)
                executed_trades[t] = executed_qty

                # Trade notification only when order executed with positive qty
                if (
                    notifier.enabled
                    and settings.notify_enabled
                    and action in ("buy", "sell", "short", "cover")
                    and executed_qty > 0
                ):
                    pos = portfolio["positions"][t]
                    net_shares = float(pos["long"] - pos["short"])
                    position_value = float(net_shares * price)
                    notifier.notify_trade(
                        ts=now_ts,
                        ticker=t,
                        action=action,
                        quantity=executed_qty,
                        price=price,
                        net_shares=net_shares,
                        position_value=position_value,
                        cash_after=portfolio["cash"],
                        min_interval_seconds=int(
                            getattr(settings, "notify_live_trade_cooldown_seconds", 0) or 0
                        ),
                    )

            # Recalculate portfolio value and exposures after executing trades
            total_val = live_bt.calculate_portfolio_value(current_prices)
            long_exp = sum(portfolio["positions"][t]["long"] * current_prices.get(t, 0.0) for t in settings.signals.tickers)
            short_exp = sum(portfolio["positions"][t]["short"] * current_prices.get(t, 0.0) for t in settings.signals.tickers)
            gross = long_exp + short_exp
            net = long_exp - short_exp
            lsr = long_exp / short_exp if short_exp > 1e-9 else float("inf")

            # Track time series like backtest for analysis
            live_bt.portfolio_values.append(
                {
                    "Date": now_ts,
                    "Portfolio Value": total_val,
                    "Long Exposure": long_exp,
                    "Short Exposure": short_exp,
                    "Gross Exposure": gross,
                    "Net Exposure": net,
                    "Long/Short Ratio": lsr,
                }
            )

            # Print concise decisions and current portfolio value
            print({"decisions": decisions, "total_value": round(total_val, 2), "cash": round(portfolio["cash"], 2)})

            # Periodic summary notification
            if notifier.enabled and settings.notify_enabled:
                if int(getattr(settings, "notify_live_summary_seconds", 0) or 0) > 0:
                    notifier.notify_summary(
                        ts=now_ts,
                        total_value=total_val,
                        cash=portfolio["cash"],
                        long_exposure=long_exp,
                        short_exposure=short_exp,
                        gross_exposure=gross,
                        net_exposure=net,
                        long_short_ratio=lsr,
                        min_interval_seconds=int(
                            getattr(settings, "notify_live_summary_seconds", 0) or 0
                        ),
                    )

            # Sleep strategy:
            # - If poll > 0, sleep fixed seconds
            # - Else, align sleep until the close of next primary_interval bar
            if poll > 0:
                time.sleep(poll)
            else:
                # Compute next bar close aligned to primary_interval
                from src.utils.constants import Interval
                import pandas as pd
                interval: Interval = settings.primary_interval
                delta: pd.Timedelta = interval.to_timedelta()
                # Current UTC time as pandas Timestamp (tz-aware)
                now_ts_utc = pd.Timestamp.now(tz="UTC")
                # Align by flooring to interval and adding one interval
                floor = now_ts_utc.floor(freq=delta)
                next_close = floor + delta
                sleep_seconds = max(1.0, (next_close.to_pydatetime() - now_ts_utc.to_pydatetime()).total_seconds())
                time.sleep(sleep_seconds)
