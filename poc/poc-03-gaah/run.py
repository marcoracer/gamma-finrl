"""POC-03: GAAH Features — add temporal markings to FinRL pipeline.

Tests whether GAAH temporal markings (validated in fenix POC-014/015)
improve PPO agent performance when used as observation features in
the FinRL StockTradingEnv.

Usage:
    uv run python poc/poc-03-gaah/run.py
    uv run python poc/poc-03-gaah/run.py --test-period 2024
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
    TICKERS_10,
    TRAIN_END,
    TRAIN_START,
    TEST_BEARISH_END,
    TEST_BEARISH_START,
    TEST_BULLISH_END,
    TEST_BULLISH_START,
)
from gamma_finrl.env_custom import DSRStockTradingEnv
from gamma_finrl.features import GAAH_INDICATORS, add_gaah_to_dataframe

# Combined indicator list
ALL_INDICATORS = INDICATORS + GAAH_INDICATORS


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


def create_env(df, tickers, initial_amount=100_000, hmax=100):
    stock_dim = len(tickers)
    env = DSRStockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=initial_amount,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        reward_scaling=1.0,
        state_space=1 + 2 * stock_dim + stock_dim * len(ALL_INDICATORS),
        action_space=stock_dim,
        tech_indicator_list=ALL_INDICATORS,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=100,
    )
    return env


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
    trained = agent.train_model(model=model, tb_log_name="ppo_gaah", total_timesteps=total_timesteps)
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

    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0.0

    peak = np.maximum.accumulate(values)
    max_drawdown = np.max((peak - values) / peak) if len(values) > 0 else 0.0

    neg = daily_returns[daily_returns < 0]
    sortino = np.mean(daily_returns) / np.sqrt(np.mean(neg**2)) * np.sqrt(252) if len(neg) > 1 else 0.0

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "final_nav": float(values[-1]),
        "n_days": int(len(values)),
    }


def main():
    parser = argparse.ArgumentParser(description="POC-03: GAAH Features")
    parser.add_argument("--test-period", default="2024", choices=["2023", "2024"])
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_start = TEST_BULLISH_START if args.test_period == "2023" else TEST_BEARISH_START
    test_end = TEST_BULLISH_END if args.test_period == "2023" else TEST_BEARISH_END

    print(f"\n{'=' * 60}")
    print(f"  POC-03: GAAH Features ({args.test_period})")
    print(f"  Indicators: {len(INDICATORS)} standard + {len(GAAH_INDICATORS)} GAAH = {len(ALL_INDICATORS)} total")
    print(f"{'=' * 60}")

    # Data
    df_raw = download_b3_data(TICKERS_10, TRAIN_START, test_end)
    df = add_features(df_raw)

    # Add GAAH features
    print("Adding GAAH temporal markings...")
    df = add_gaah_to_dataframe(df, TICKERS_10, L=5, K=0.5, method="triangle")
    df = df.dropna(subset=ALL_INDICATORS).reset_index(drop=True)
    print(f"  Total features: {len(ALL_INDICATORS)}, rows: {len(df)}")

    train_df = data_split(df, TRAIN_START, TRAIN_END)
    test_df = data_split(df, test_start, test_end)

    # Env
    train_env = create_env(train_df, TICKERS_10)
    test_env = create_env(test_df, TICKERS_10)

    # Train
    model, train_time = train_ppo(train_env, total_timesteps=args.timesteps, seed=args.seed)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(model, test_env)

    regime = "bullish" if args.test_period == "2023" else "bearish"
    print(f"\n{'=' * 60}")
    print(f"  Results — {args.test_period} ({regime}), DSR + GAAH")
    print(f"{'=' * 60}")
    print(f"  Total Return:   {metrics['total_return']:+.2%}")
    print(f"  Sharpe Ratio:   {metrics['sharpe']:.4f}")
    print(f"  Sortino Ratio:  {metrics['sortino']:.4f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2%}")
    print(f"  Final NAV:      R$ {metrics['final_nav']:,.2f}")

    # Save
    out_dir = ROOT / "results" / "poc-03"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "poc": "poc-03-gaah",
        "test_period": args.test_period,
        "regime": regime,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "n_indicators": len(ALL_INDICATORS),
        "train_time_s": train_time,
        "metrics": metrics,
    }
    with open(out_dir / f"result_{args.test_period}_seed{args.seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {out_dir}")


if __name__ == "__main__":
    main()
