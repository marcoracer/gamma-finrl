"""Configuration for Gamma FinRL experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


# B3 tickers — same universe as gamma-fenix
TICKERS_10 = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "BBAS3.SA",
    "ABEV3.SA",
    "WEGE3.SA",
    "RENT3.SA",
    "SUZB3.SA",
    "EQTL3.SA",
]

TICKERS_20 = TICKERS_10 + [
    "B3SA3.SA",
    "RADL3.SA",
    "SBSP3.SA",
    "PRIO3.SA",
    "GGBR4.SA",
    "RAIL3.SA",
    "TOTS3.SA",
    "BBSE3.SA",
    "UGPA3.SA",
    "BPAC11.SA",
]

BENCHMARK = "^BVSP"

# B3 transaction costs (ADR-011)
B3_BUY_COST_PCT = 0.001  # ~0.1% round-trip
B3_SELL_COST_PCT = 0.001

# Date ranges matching gamma-fenix POCs
TRAIN_START = "2019-01-02"
TRAIN_END = "2022-12-30"
TEST_BULLISH_START = "2023-01-02"
TEST_BULLISH_END = "2023-12-29"
TEST_BEARISH_START = "2024-01-02"
TEST_BEARISH_END = "2024-12-31"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # Tickers
    tickers: list[str] = field(default_factory=lambda: TICKERS_10)

    # Dates
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    test_start: str = TEST_BULLISH_START
    test_end: str = TEST_BULLISH_END

    # Environment
    initial_capital: float = 100_000.0
    hmax: int = 100  # max shares per trade
    buy_cost_pct: float = B3_BUY_COST_PCT
    sell_cost_pct: float = B3_SELL_COST_PCT
    reward_scaling: float = 1.0

    # PPO
    algorithm: str = "ppo"
    total_timesteps: int = 100_000
    seed: int = 42
    net_arch: list[int] = field(default_factory=lambda: [256, 128])
    learning_rate: float = 3e-4

    # Features
    tech_indicators: list[str] = field(default_factory=lambda: [
        "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30",
        "close_30_sma", "close_60_sma",
    ])

    # Logging
    experiment_name: str = "default"
    log_dir: str = "results"
