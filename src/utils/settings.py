from pydantic_settings import BaseSettings
from pydantic import model_validator, BaseModel
from datetime import datetime
import yaml
from typing import List, Optional
from dotenv import load_dotenv
from .constants import Interval

load_dotenv()


class SignalSettings(BaseModel):
    intervals: List[Interval]
    tickers: List[str]
    strategies: List[str]


class ModelSettings(BaseModel):
    name: str
    provider: str
    base_url: Optional[str] = None


class Settings(BaseSettings):
    mode: str
    start_date: datetime
    end_date: datetime
    primary_interval: Interval
    initial_cash: int
    margin_requirement: float
    show_reasoning: bool
    show_agent_graph: bool = True
    signals: SignalSettings
    model: ModelSettings
    # live options
    live_poll_seconds: Optional[int] = 60  # default 60s when live; 0/None -> run once
    notify_every_steps: Optional[int] = 1  # backtest notifications frequency
    # notifications
    notify_enabled: bool = True
    notify_skip_hold: bool = True
    notify_live_trade_cooldown_seconds: int = 300  # send same trade per ticker at most every N seconds
    notify_live_summary_seconds: int = 0  # 0 disables live summary notifications

    @model_validator(mode='after')
    def check_primary_interval_in_intervals(self):
        if self.primary_interval not in self.signals.intervals:
            raise ValueError(
                f"primary_interval '{self.primary_interval}' must be in signals.intervals {self.signals.intervals}")
        return self


def load_settings(yaml_path: str = "config.yaml") -> Settings:
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return Settings(**yaml_data)


# Load and use
settings = load_settings()

# print(settings.model.name)
# print(settings.model.provider)
# print(settings.mode)
# print(settings.primary_interval)
# print(settings.start_date)
# print(settings.end_date)
# print(settings.signals)
