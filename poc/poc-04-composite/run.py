"""POC-04: Composite Score as Reward.

Uses the gamma-fenix composite_score (Sortino 40% + Alpha 30% + Calmar 30%)
as the step reward instead of DSR or delta_NAV.

Reward = delta_composite (change in rolling composite score from last step).

The composite score requires accumulating returns and benchmark returns
over a rolling window, then computing Sortino, Alpha, Calmar, and MDD
at each step.

Usage:
    uv run python poc/poc-04-composite/run.py
    uv run python poc/poc-04-composite/run.py --test-period 2024
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

from gamma_finrl.config import (
    B3_BUY_COST_PCT,
    TICKERS_10,
    TRAIN_END,
    TRAIN_START,
    TEST_BEARISH_END,
    TEST_BEARISH_START,
    TEST_BULLISH_END,
    TEST_BULLISH_START,
)


# ── Composite Reward State ───────────────────────────────────────────────


def _sigmoid(x: float, k: float = 1.0) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-k * x))
    except (OverflowError, ValueError):
        return 0.0 if x < 0 else 1.0


def composite_score(sortino: float, alpha: float, calmar: float, max_drawdown: float, sharpe: float = 0.0) -> float:
    """Composite fitness score — same as gamma-fenix domain/metrics.py."""
    sortino_norm = _sigmoid(sortino - 1.0, k=1.0)
    alpha_norm = _sigmoid(alpha - 0.1, k=5.0)
    calmar_norm = _sigmoid(calmar - 1.0, k=1.0)

    raw = 0.4 * sortino_norm + 0.3 * alpha_norm + 0.3 * calmar_norm

    if sharpe < 0:
        sharpe_penalty = _sigmoid(-sharpe, k=2.0) * 0.7
        raw *= 1.0 - sharpe_penalty

    if max_drawdown > 0.5:
        excess = (max_drawdown - 0.5) / 0.5
        penalty = 1.0 - 0.8 * min(excess, 1.0) ** 0.5
        raw *= penalty

    return raw


class CompositeRewardState:
    """Tracks rolling metrics and computes composite score incrementally."""

    def __init__(self, window: int = 60):
        self.window = window
        self.returns: deque[float] = deque(maxlen=window)
        self.benchmark_returns: deque[float] = deque(maxlen=window)
        self.nav_history: list[float] = []
        self._prev_score: float = 0.0

    def reset(self) -> None:
        self.returns.clear()
        self.benchmark_returns.clear()
        self.nav_history.clear()
        self._prev_score = 0.0

    def update(
        self,
        portfolio_return: float,
        benchmark_return: float,
        nav: float,
    ) -> float:
        """Update state and return delta composite score as reward."""
        self.returns.append(portfolio_return)
        self.benchmark_returns.append(benchmark_return)
        self.nav_history.append(nav)

        # Need at least a few data points for meaningful metrics
        if len(self.returns) < 5:
            return 0.0

        rets = np.array(self.returns)
        bench_rets = np.array(self.benchmark_returns)
        navs = np.array(self.nav_history[-len(rets):])

        # Sharpe
        if len(rets) > 1 and np.std(rets) > 0:
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252))
        else:
            sharpe = 0.0

        # Sortino (downside deviation)
        neg_rets = rets[rets < 0]
        if len(neg_rets) > 1:
            downside_std = float(np.sqrt(np.mean(neg_rets**2)))
            sortino = float(np.mean(rets) / downside_std * np.sqrt(252))
        else:
            sortino = 0.0

        # Alpha (cumulative return vs benchmark)
        port_cum = float(np.prod(1 + rets) - 1)
        bench_cum = float(np.prod(1 + bench_rets) - 1)
        alpha = port_cum - bench_cum

        # Max drawdown
        if len(navs) > 1:
            peak = np.maximum.accumulate(navs)
            dd = (peak - navs) / np.maximum(peak, 1e-10)
            mdd = float(np.max(dd))
        else:
            mdd = 0.0

        # Calmar (annualized return / MDD)
        n_days = len(rets)
        annual_return = port_cum * (252 / max(n_days, 1))
        calmar = annual_return / mdd if mdd > 0.001 else 0.0

        # Compute composite score
        score = composite_score(sortino, alpha, calmar, mdd, sharpe=sharpe)

        # Reward = delta composite
        reward = score - self._prev_score
        self._prev_score = score

        return reward


# ── Custom Environment ───────────────────────────────────────────────────


class CompositeRewardEnv(StockTradingEnv):
    """StockTradingEnv with composite score as reward."""

    def __init__(self, *args, benchmark_returns: np.ndarray | None = None, window: int = 60, **kwargs):
        super().__init__(*args, **kwargs)
        self._benchmark_returns = benchmark_returns
        self._composite_state = CompositeRewardState(window=window)
        self._step_count = 0

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._composite_state.reset()
        self._step_count = 0
        return result

    def step(self, actions):
        result = super().step(actions)

        state = result[0]
        old_reward = result[1]
        terminated = result[2]
        truncated = result[3]
        info = result[4] if len(result) > 4 else {}

        # Portfolio return from NAV change
        old_nav = self.asset_memory[-2] if len(self.asset_memory) >= 2 else self.initial_amount
        new_nav = self.asset_memory[-1] if self.asset_memory else self.initial_amount
        portfolio_return = (new_nav - old_nav) / max(old_nav, 1e-10)

        # Benchmark return for this step
        bench_ret = 0.0
        if self._benchmark_returns is not None and self._step_count < len(self._benchmark_returns):
            bench_ret = float(self._benchmark_returns[self._step_count])

        self._step_count += 1

        # Composite reward
        reward = self._composite_state.update(portfolio_return, bench_ret, new_nav)
        reward = reward * self.reward_scaling

        return state, reward, terminated, truncated, info


# ── Data ─────────────────────────────────────────────────────────────────


def download_b3_data(tickers, start, end):
    import yfinance as yf
    print(f"Downloading {len(tickers)} tickers: {start} to {end}...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1, future_stack=True).reset_index()
    else:
        df = df.reset_index()
        df["Ticker"] = tickers[0]
    df = df.rename(columns={
        "Date": "date", "Ticker": "tic", "Open": "open",
        "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df = df.dropna(subset=["close", "open", "high", "low", "volume"])
    print(f"  Downloaded {len(df)} rows, {df['tic'].nunique()} tickers")
    return df


def add_features(df):
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df = fe.preprocess_data(df)
    df = df.dropna(subset=INDICATORS).reset_index(drop=True)
    return df


def get_benchmark_returns(start, end):
    """Download IBOV returns for benchmark."""
    import yfinance as yf
    bench = yf.download("^BVSP", start=start, end=end, auto_adjust=True)
    if isinstance(bench.columns, pd.MultiIndex):
        bench = bench.droplevel(1, axis=1)
    close = bench["Close"]
    returns = close.pct_change().dropna().values
    print(f"  Benchmark returns: {len(returns)} days")
    return returns


def create_env(df, tickers, benchmark_returns=None, initial_amount=100_000, hmax=100):
    stock_dim = len(tickers)
    env = CompositeRewardEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        reward_scaling=100.0,  # scale up small delta-composite values
        state_space=1 + 2 * stock_dim + stock_dim * len(INDICATORS),
        action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=100,
        benchmark_returns=benchmark_returns,
        window=60,
    )
    return env


# ── Training & Evaluation ────────────────────────────────────────────────


def train_ppo(env, total_timesteps=100_000, seed=42):
    agent = DRLAgent(env=env)
    model = agent.get_model(
        "ppo",
        policy_kwargs={"net_arch": [256, 128]},
        model_kwargs={
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
        },
        seed=seed,
        verbose=0,
    )
    print(f"Training PPO for {total_timesteps} timesteps (seed={seed})...")
    t0 = time.perf_counter()
    trained = agent.train_model(model=model, tb_log_name="ppo_composite", total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    return trained, elapsed


def evaluate(model, env):
    account_values, _ = DRLAgent.DRL_prediction(model=model, environment=env)
    if isinstance(account_values, pd.DataFrame):
        values = account_values["account_value"].values
    else:
        values = np.array(account_values)

    total_return = (values[-1] - values[0]) / values[0]
    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0.0

    peak = np.maximum.accumulate(values)
    max_drawdown = float(np.max((peak - values) / peak)) if len(values) > 0 else 0.0

    neg = daily_returns[daily_returns < 0]
    sortino = float(np.mean(daily_returns) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 1 else 0.0

    annual_return = total_return * (252 / max(len(values) - 1, 1))
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_drawdown),
        "final_nav": float(values[-1]),
        "n_days": int(len(values)),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="POC-04: Composite Reward")
    parser.add_argument("--test-period", default="2023", choices=["2023", "2024"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_start = TEST_BULLISH_START if args.test_period == "2023" else TEST_BEARISH_START
    test_end = TEST_BULLISH_END if args.test_period == "2023" else TEST_BEARISH_END

    print(f"\n{'=' * 60}")
    print(f"  POC-04: Composite Score as Reward ({args.test_period})")
    print(f"  Components: Sortino(40%) + Alpha(30%) + Calmar(30%)")
    print(f"  Reward = delta_composite × 100 (rolling window=60d)")
    print(f"{'=' * 60}")

    # Data
    df_raw = download_b3_data(TICKERS_10, TRAIN_START, test_end)
    df = add_features(df_raw)
    train_df = data_split(df, TRAIN_START, TRAIN_END)
    test_df = data_split(df, test_start, test_end)

    # Benchmark returns
    train_bench = get_benchmark_returns(TRAIN_START, TRAIN_END)
    test_bench = get_benchmark_returns(test_start, test_end)

    # Environments
    train_env = create_env(train_df, TICKERS_10, benchmark_returns=train_bench)
    test_env = create_env(test_df, TICKERS_10, benchmark_returns=test_bench)

    # Train
    model, train_time = train_ppo(train_env, total_timesteps=args.timesteps, seed=args.seed)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(model, test_env)

    regime = "bullish" if args.test_period == "2023" else "bearish"
    print(f"\n{'=' * 60}")
    print(f"  Results — {args.test_period} ({regime}), Composite Reward")
    print(f"{'=' * 60}")
    print(f"  Total Return:   {metrics['total_return']:+.2%}")
    print(f"  Sharpe Ratio:   {metrics['sharpe']:.4f}")
    print(f"  Sortino Ratio:  {metrics['sortino']:.4f}")
    print(f"  Calmar Ratio:   {metrics['calmar']:.4f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"  Final NAV:      R$ {metrics['final_nav']:,.2f}")

    # Save
    out_dir = ROOT / "results" / "poc-04"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "poc": "poc-04-composite-reward",
        "test_period": args.test_period,
        "regime": regime,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "reward_type": "composite_score_delta",
        "train_time_s": train_time,
        "metrics": metrics,
    }
    with open(out_dir / f"result_{args.test_period}_seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_dir}")


if __name__ == "__main__":
    main()
