import os
import requests
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta


class TelegramNotifier:
    """
    Simple Telegram notifier. Enable by setting TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment.
    Optionally set TELEGRAM_PARSE_MODE to 'Markdown' or 'HTML' (default: Markdown).
    """

    def __init__(self) -> None:
        self.token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url: str = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        self.parse_mode: str = os.getenv("TELEGRAM_PARSE_MODE", "Markdown")
        # internal dedup/cache
        self._last_trade_sent_at: Dict[str, datetime] = {}
        self._last_summary_epoch: Optional[float] = None

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def _send(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": self.parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
        except Exception as e:
            # Fail silently in backtests, just print once
            print(f"[TelegramNotifier] send failed: {e}")

    def notify_trade(
        self,
        *,
        ts: datetime,
        ticker: str,
        action: str,
        quantity: float,
        price: float,
        net_shares: float,
        position_value: float,
        cash_after: float,
        min_interval_seconds: int = 0,
    ) -> None:
        # cooldown to avoid spamming identical tickers too frequently
        if min_interval_seconds > 0:
            last = self._last_trade_sent_at.get(ticker)
            if last and ts - last < timedelta(seconds=min_interval_seconds):
                return
        msg = (
            f"*Trade Executed*\n"
            f"Time: `{ts}`\n"
            f"Ticker: `{ticker}`\n"
            f"Action: *{action.upper()}*\n"
            f"Quantity: `{quantity}` at Price: `${price:,.2f}`\n"
            f"Net Shares: `{net_shares}`\n"
            f"Position Value: `${position_value:,.2f}`\n"
            f"Cash: `${cash_after:,.2f}`"
        )
        self._send(msg)
        self._last_trade_sent_at[ticker] = ts

    def notify_summary(
        self,
        *,
        ts: datetime,
        total_value: float,
        cash: float,
        long_exposure: float,
        short_exposure: float,
        gross_exposure: float,
        net_exposure: float,
        long_short_ratio: float,
        min_interval_seconds: int = 0,
    ) -> None:
        if min_interval_seconds > 0:
            import time
            now_epoch = time.time()
            last = self._last_summary_epoch
            if last and (now_epoch - last) < min_interval_seconds:
                return
        msg = (
            f"*Portfolio Summary*\n"
            f"Time: `{ts}`\n"
            f"Total Value: `${total_value:,.2f}`\n"
            f"Cash: `${cash:,.2f}`\n"
            f"Long Exp: `${long_exposure:,.2f}` | Short Exp: `${short_exposure:,.2f}`\n"
            f"Gross: `${gross_exposure:,.2f}` | Net: `${net_exposure:,.2f}`\n"
            f"Long/Short Ratio: `{long_short_ratio:.2f}`"
        )
        self._send(msg)
        if min_interval_seconds > 0:
            self._last_summary_epoch = now_epoch


