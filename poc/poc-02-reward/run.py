"""POC-02: StockTradingEnv with DSR reward + B3 real costs + HHI penalty.

Subclasses FinRL's StockTradingEnv to replace the default reward
(delta_total_asset) with our validated DSR reward from gamma-fenix.

Usage:
    uv run python poc/poc-02-reward/run.py
    uv run python poc/poc-02-reward/run.py --test-period 2024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    B3_SELL_COST_PCT,
    TICKERS_10,
    TRAIN_END,
    TRAIN_START,
    TEST_BEARISH_END,
    TEST_BEARISH_START,
    TEST_BULLISH_END,
    TEST_BULLISH_START,
)
from gamma_finrl.env_custom import DSRStockTradingEnv


# ── Data functions (same as POC-01) ──────────────────────────────────────


def download_b3_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download B3 data via yfinance and format for FinRL."""
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


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators via FinRL FeatureEngineer."""
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


def create_env(
    df: pd.DataFrame,
    tickers: list[str],
    initial_amount: float = 100_000,
    hmax: int = 100,
    use_dsr: bool = True,
    cost_weight: float = 1.0,
    concentration_weight: float = 0.0,
) -> StockTradingEnv:
    """Create environment — DSR or vanilla."""
    stock_dim = len(tickers)

    env_cls = DSRStockTradingEnv if use_dsr else StockTradingEnv
    extra_kwargs = {}
    if use_dsr:
        extra_kwargs = {
            "cost_weight": cost_weight,
            "concentration_weight": concentration_weight,
        }

    env = env_cls(
        df=df,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        reward_scaling=1.0,
        state_space=1 + 2 * stock_dim + stock_dim * len(INDICATORS),
        action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=100,
        **extra_kwargs,
    )
    return env


# ── Training & Evaluation (same as POC-01) ───────────────────────────────


def train_ppo(env, total_timesteps=100_000, seed=42, net_arch=None, learning_rate=3e-4):
    """Train PPO agent."""
    if net_arch is None:
        net_arch = [256, 128]

    agent = DRLAgent(env=env)
    model = agent.get_model(
        "ppo",
        policy_kwargs={"net_arch": net_arch},
        model_kwargs={
            "learning_rate": learning_rate,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
        },
        seed=seed,
        verbose=0,
    )

    print(f"Training PPO {net_arch} for {total_timesteps} timesteps (seed={seed})...")
    t0 = time.perf_counter()
    trained = agent.train_model(model=model, tb_log_name="ppo_dsr", total_timesteps=total_timesteps)
    elapsed = time.perf_counter() - t0
    print(f"  Training complete in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    return trained, elapsed


def evaluate(model, env):
    """Run backtest and compute metrics."""
    account_values, actions = DRLAgent.DRL_prediction(model=model, environment=env)

    if isinstance(account_values, pd.DataFrame):
        values = account_values["account_value"].values
    else:
        values = np.array(account_values)

    initial = values[0]
    final = values[-1]
    total_return = (final - initial) / initial

    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

    neg_returns = daily_returns[daily_returns < 0]
    if len(neg_returns) > 1:
        downside_std = np.sqrt(np.mean(neg_returns**2))
        sortino = np.mean(daily_returns) / downside_std * np.sqrt(252)
    else:
        sortino = 0.0

    annual_return = total_return * (252 / max(len(values) - 1, 1))
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_drawdown),
        "final_nav": float(final),
        "n_days": int(len(values)),
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="POC-02: DSR Reward")
    parser.add_argument("--test-period", default="2023", choices=["2023", "2024"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--concentration-weight", type=float, default=0.0)
    args = parser.parse_args()

    test_start = TEST_BULLISH_START if args.test_period == "2023" else TEST_BEARISH_START
    test_end = TEST_BULLISH_END if args.test_period == "2023" else TEST_BEARISH_END

    print(f"\n{'=' * 60}")
    print(f"  POC-02: DSR Reward ({args.test_period})")
    print(f"  cost_weight={args.cost_weight}, conc_weight={args.concentration_weight}")
    print(f"{'=' * 60}")

    # Data
    df_raw = download_b3_data(TICKERS_10, TRAIN_START, test_end)
    df = add_features(df_raw)
    train_df = data_split(df, TRAIN_START, TRAIN_END)
    test_df = data_split(df, test_start, test_end)

    # Environments — DSR reward
    train_env = create_env(
        train_df, TICKERS_10, hmax=100,
        use_dsr=True,
        cost_weight=args.cost_weight,
        concentration_weight=args.concentration_weight,
    )
    test_env = create_env(
        test_df, TICKERS_10, hmax=100,
        use_dsr=True,
        cost_weight=args.cost_weight,
        concentration_weight=args.concentration_weight,
    )

    # Train
    model, train_time = train_ppo(train_env, total_timesteps=args.timesteps, seed=args.seed)

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate(model, test_env)

    regime = "bullish" if args.test_period == "2023" else "bearish"
    print(f"\n{'=' * 60}")
    print(f"  Results — {args.test_period} ({regime}), DSR reward")
    print(f"{'=' * 60}")
    print(f"  Total Return:   {metrics['total_return']:+.2%}")
    print(f"  Sharpe Ratio:   {metrics['sharpe']:.4f}")
    print(f"  Sortino Ratio:  {metrics['sortino']:.4f}")
    print(f"  Calmar Ratio:   {metrics['calmar']:.4f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"  Final NAV:      R$ {metrics['final_nav']:,.2f}")

    # Save
    out_dir = ROOT / "results" / "poc-02"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "poc": "poc-02-dsr-reward",
        "test_period": args.test_period,
        "regime": regime,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "cost_weight": args.cost_weight,
        "concentration_weight": args.concentration_weight,
        "train_time_s": train_time,
        "metrics": metrics,
    }
    with open(out_dir / f"result_{args.test_period}_seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_dir}")

    return metrics


if __name__ == "__main__":
    main()
