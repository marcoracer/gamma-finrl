"""POC-05: Optuna Hyperparameter Tuning with Composite Reward.

Uses Optuna TPE sampler + MedianPruner to search PPO hyperparameters.
Objective: maximize composite score on a validation period.

Search space: learning_rate, net_arch, n_steps, batch_size, n_epochs,
              reward_scaling (for composite reward).

Usage:
    uv run python poc/poc-05-optuna/run.py --n-trials 20
    uv run python poc/poc-05-optuna/run.py --n-trials 50 --timesteps 50000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS
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


# ── Composite Reward (same as POC-04) ────────────────────────────────────


def _sigmoid(x, k=1.0):
    try:
        return 1.0 / (1.0 + math.exp(-k * x))
    except (OverflowError, ValueError):
        return 0.0 if x < 0 else 1.0


def composite_score(sortino, alpha, calmar, max_drawdown, sharpe=0.0):
    sortino_norm = _sigmoid(sortino - 1.0, k=1.0)
    alpha_norm = _sigmoid(alpha - 0.1, k=5.0)
    calmar_norm = _sigmoid(calmar - 1.0, k=1.0)
    raw = 0.4 * sortino_norm + 0.3 * alpha_norm + 0.3 * calmar_norm
    if sharpe < 0:
        raw *= 1.0 - _sigmoid(-sharpe, k=2.0) * 0.7
    if max_drawdown > 0.5:
        excess = (max_drawdown - 0.5) / 0.5
        raw *= 1.0 - 0.8 * min(excess, 1.0) ** 0.5
    return raw


class CompositeRewardState:
    def __init__(self, window=60):
        self.window = window
        self.returns: deque[float] = deque(maxlen=window)
        self.benchmark_returns: deque[float] = deque(maxlen=window)
        self.nav_history: list[float] = []
        self._prev_score = 0.0

    def reset(self):
        self.returns.clear()
        self.benchmark_returns.clear()
        self.nav_history.clear()
        self._prev_score = 0.0

    def update(self, portfolio_return, benchmark_return, nav):
        self.returns.append(portfolio_return)
        self.benchmark_returns.append(benchmark_return)
        self.nav_history.append(nav)
        if len(self.returns) < 5:
            return 0.0
        rets = np.array(self.returns)
        bench_rets = np.array(self.benchmark_returns)
        navs = np.array(self.nav_history[-len(rets):])

        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0.0
        neg = rets[rets < 0]
        sortino = float(np.mean(rets) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 1 else 0.0
        port_cum = float(np.prod(1 + rets) - 1)
        bench_cum = float(np.prod(1 + bench_rets) - 1)
        alpha = port_cum - bench_cum
        if len(navs) > 1:
            peak = np.maximum.accumulate(navs)
            dd = (peak - navs) / np.maximum(peak, 1e-10)
            mdd = float(np.max(dd))
        else:
            mdd = 0.0
        annual_return = port_cum * (252 / max(len(rets), 1))
        calmar = annual_return / mdd if mdd > 0.001 else 0.0

        score = composite_score(sortino, alpha, calmar, mdd, sharpe=sharpe)
        reward = score - self._prev_score
        self._prev_score = score
        return reward


class CompositeRewardEnv(gym.Env):
    """Wrapper around FinRL StockTradingEnv with composite reward."""

    metadata = {"render_modes": []}

    def __init__(self, finrl_env, benchmark_returns=None, reward_scaling=100.0, window=60):
        super().__init__()
        self._env = finrl_env
        self._benchmark_returns = benchmark_returns
        self._reward_scaling = reward_scaling
        self._composite_state = CompositeRewardState(window=window)
        self._step_count = 0
        # SB3 needs these on the env itself
        self.observation_space = finrl_env.observation_space
        self.action_space = finrl_env.action_space

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        self._composite_state.reset()
        self._step_count = 0
        return result

    def step(self, actions):
        result = self._env.step(actions)
        state = result[0]
        terminated = result[2]
        truncated = result[3]
        info = result[4] if len(result) > 4 else {}

        old_nav = self._env.asset_memory[-2] if len(self._env.asset_memory) >= 2 else self._env.initial_amount
        new_nav = self._env.asset_memory[-1] if self._env.asset_memory else self._env.initial_amount
        portfolio_return = (new_nav - old_nav) / max(old_nav, 1e-10)

        bench_ret = 0.0
        if self._benchmark_returns is not None and self._step_count < len(self._benchmark_returns):
            bench_ret = float(self._benchmark_returns[self._step_count])
        self._step_count += 1

        reward = self._composite_state.update(portfolio_return, bench_ret, new_nav)
        reward = reward * self._reward_scaling
        return state, reward, terminated, truncated, info

    def close(self):
        pass


# ── Data ─────────────────────────────────────────────────────────────────

_cached_data = None
_cached_bench = None


def load_data(test_end):
    global _cached_data, _cached_bench
    if _cached_data is not None and _cached_data.get("test_end") == test_end:
        return _cached_data, _cached_bench

    import yfinance as yf
    print("Downloading data...")
    df = yf.download(TICKERS_10, start=TRAIN_START, end=test_end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1, future_stack=True).reset_index()
    else:
        df = df.reset_index()
        df["Ticker"] = TICKERS_10[0]
    df = df.rename(columns={
        "Date": "date", "Ticker": "tic", "Open": "open",
        "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df = df.dropna(subset=["close", "open", "high", "low", "volume"])

    fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS, use_vix=False, use_turbulence=False, user_defined_feature=False)
    df = fe.preprocess_data(df)
    df = df.dropna(subset=INDICATORS).reset_index(drop=True)

    # Benchmark
    bench = yf.download("^BVSP", start=TRAIN_START, end=test_end, auto_adjust=True)
    if isinstance(bench.columns, pd.MultiIndex):
        bench = bench.droplevel(1, axis=1)
    bench_close = bench["Close"]
    bench_rets_all = bench_close.pct_change().dropna()
    dates = bench_rets_all.index

    train_df = data_split(df, TRAIN_START, TRAIN_END)
    val_df = data_split(df, TEST_BULLISH_START, TEST_BULLISH_END)
    test_df = data_split(df, TEST_BEARISH_START, TEST_BEARISH_END)

    train_bench = bench_rets_all[(dates >= TRAIN_START) & (dates <= TRAIN_END)].values
    val_bench = bench_rets_all[(dates >= TEST_BULLISH_START) & (dates <= TEST_BULLISH_END)].values
    test_bench = bench_rets_all[(dates >= TEST_BEARISH_START) & (dates <= TEST_BEARISH_END)].values

    data = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "test_end": test_end,
    }
    bench_data = {
        "train": train_bench,
        "val": val_bench,
        "test": test_bench,
    }
    _cached_data = data
    _cached_bench = bench_data
    print("  Data loaded and cached")
    return data, bench_data


def make_finrl_env(df, hmax=100):
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    stock_dim = len(TICKERS_10)
    return StockTradingEnv(
        df=df, stock_dim=stock_dim, hmax=hmax, initial_amount=100_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        sell_cost_pct=[B3_BUY_COST_PCT] * stock_dim,
        reward_scaling=1.0,
        state_space=1 + 2 * stock_dim + stock_dim * len(INDICATORS),
        action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        turbulence_threshold=None, make_plots=False, print_verbosity=999,
    )


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(model, env, bench_returns):
    """Run backtest by stepping through env manually (avoids FinRL's get_sb_env)."""
    obs, _ = env.reset()
    done = False
    values = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        result = env.step(action)
        obs = result[0]
        done = result[2] or result[3]
        # Track NAV from inner FinRL env
        if hasattr(env, '_env') and hasattr(env._env, 'asset_memory'):
            values.append(env._env.asset_memory[-1])

    if len(values) < 2:
        return {"total_return": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0,
                "max_drawdown": 0.0, "final_nav": 100_000.0, "benchmark_return": 0.0,
                "alpha": 0.0, "composite_score": 0.0}

    values = np.array(values)

    total_return = (values[-1] - values[0]) / values[0]
    daily_returns = np.diff(values) / values[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0.0
    neg = daily_returns[daily_returns < 0]
    sortino = float(np.mean(daily_returns) / np.sqrt(np.mean(neg**2)) * np.sqrt(252)) if len(neg) > 1 else 0.0
    peak = np.maximum.accumulate(values)
    mdd = float(np.max((peak - values) / peak)) if len(values) > 0 else 0.0
    annual_return = total_return * (252 / max(len(values) - 1, 1))
    calmar = annual_return / mdd if mdd > 0 else 0.0

    # Benchmark return
    bench_return = float(np.prod(1 + bench_returns[:len(daily_returns)]) - 1) if len(bench_returns) > 0 else 0.0
    alpha = total_return - bench_return

    cs = composite_score(sortino, alpha, calmar, mdd, sharpe=sharpe)

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(mdd),
        "final_nav": float(values[-1]),
        "benchmark_return": float(bench_return),
        "alpha": float(alpha),
        "composite_score": float(cs),
    }


# ── Optuna Objective ─────────────────────────────────────────────────────


def create_objective(data, bench_data, timesteps):
    def objective(trial: optuna.Trial) -> float:
        from stable_baselines3 import PPO

        # Search space
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        widths = trial.suggest_categorical("layer_width", [64, 128, 256, 512])
        net_arch = [widths] * n_layers
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20, 30])
        reward_scaling = trial.suggest_float("reward_scaling", 10.0, 500.0, log=True)
        window = trial.suggest_categorical("window", [30, 60, 90])

        # Create envs
        finrl_train = make_finrl_env(data["train_df"])
        train_env = CompositeRewardEnv(
            finrl_train, benchmark_returns=bench_data["train"],
            reward_scaling=reward_scaling, window=window,
        )

        # Train
        try:
            model = PPO(
                "MlpPolicy", train_env,
                policy_kwargs={"net_arch": net_arch},
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                seed=42,
                verbose=0,
                device="cpu",
            )
            model.learn(total_timesteps=timesteps)
        except Exception as e:
            print(f"  Trial {trial.number} FAILED: {e}")
            return -999.0
        finally:
            train_env.close()

        # Evaluate on validation (2023 bullish)
        finrl_val = make_finrl_env(data["val_df"])
        val_env = CompositeRewardEnv(
            finrl_val, benchmark_returns=bench_data["val"],
            reward_scaling=reward_scaling, window=window,
        )

        try:
            metrics = evaluate_model(model, val_env, bench_data["val"])
        except Exception as e:
            print(f"  Trial {trial.number} EVAL FAILED: {e}")
            return -999.0
        finally:
            val_env.close()

        trial.set_user_attr("metrics_2023", metrics)
        trial.set_user_attr("net_arch", str(net_arch))

        print(f"  Trial {trial.number}: composite={metrics['composite_score']:.4f} "
              f"ret={metrics['total_return']:+.2%} sharpe={metrics['sharpe']:.4f} "
              f"alpha={metrics['alpha']:+.2%} arch={net_arch} lr={lr:.1e}")

        return metrics["composite_score"]

    return objective


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="POC-05: Optuna Tuning")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=50_000)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  POC-05: Optuna Hyperparameter Tuning")
    print(f"  {args.n_trials} trials × {args.timesteps} timesteps")
    print(f"  Objective: maximize composite score on 2023 validation")
    print(f"{'=' * 60}")

    data, bench_data = load_data(TEST_BEARISH_END)

    # Run Optuna
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    objective = create_objective(data, bench_data, args.timesteps)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # Best trial
    best = study.best_trial
    print(f"\n{'=' * 60}")
    print(f"  BEST TRIAL: #{best.number}")
    print(f"{'=' * 60}")
    print(f"  Composite Score: {best.value:.6f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    metrics_2023 = best.user_attrs.get("metrics_2023", {})

    # Retrain best config and evaluate on BOTH periods
    print(f"\nRetraining best config on full budget (100k steps)...")

    best_lr = best.params["learning_rate"]
    best_arch = [best.params["layer_width"]] * best.params["n_layers"]
    best_reward_scaling = best.params["reward_scaling"]
    best_window = best.params["window"]

    from stable_baselines3 import PPO

    finrl_train = make_finrl_env(data["train_df"])
    train_env = CompositeRewardEnv(
        finrl_train, benchmark_returns=bench_data["train"],
        reward_scaling=best_reward_scaling, window=best_window,
    )

    model = PPO(
        "MlpPolicy", train_env,
        policy_kwargs={"net_arch": best_arch},
        learning_rate=best_lr,
        n_steps=best.params["n_steps"],
        batch_size=best.params["batch_size"],
        n_epochs=best.params["n_epochs"],
        seed=42, verbose=0, device="cpu",
    )
    model.learn(total_timesteps=100_000)
    train_env.close()

    # Eval 2023
    finrl_val = make_finrl_env(data["val_df"])
    val_env = CompositeRewardEnv(finrl_val, benchmark_returns=bench_data["val"], reward_scaling=best_reward_scaling, window=best_window)
    m2023 = evaluate_model(model, val_env, bench_data["val"])
    val_env.close()

    # Eval 2024
    finrl_test = make_finrl_env(data["test_df"])
    test_env = CompositeRewardEnv(finrl_test, benchmark_returns=bench_data["test"], reward_scaling=best_reward_scaling, window=best_window)
    m2024 = evaluate_model(model, test_env, bench_data["test"])
    test_env.close()

    # ── Final Table ──────────────────────────────────────────────────
    bench_2023 = float(np.prod(1 + bench_data["val"]) - 1)
    bench_2024 = float(np.prod(1 + bench_data["test"]) - 1)

    # Normalize metrics to consistent keys: total_return, sharpe, sortino, calmar, max_drawdown
    def _norm(m):
        return {
            "total_return": m.get("total_return", m.get("return", 0.0)),
            "sharpe": m.get("sharpe", 0.0),
            "sortino": m.get("sortino", 0.0),
            "calmar": m.get("calmar", 0.0),
            "max_drawdown": m.get("max_drawdown", m.get("mdd", 0.0)),
        }

    print(f"\n{'=' * 120}")
    print(f"  FINAL RESULTS — Optuna Best Config vs All POCs")
    print(f"{'=' * 120}")
    print(f"{'Config':<25} {'Period':<6} {'Return':>8} {'IBOV':>8} {'Alpha':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MDD':>7} {'Composite':>10}")
    print('-' * 120)

    # All results for comparison (using normalized keys)
    all_results = {
        "POC-01 Vanilla": {
            "2023": _norm({"return": 0.2391, "sharpe": 1.5379, "sortino": 1.7309, "calmar": 3.4620, "mdd": 0.0705}),
            "2024": _norm({"return": -0.0914, "sharpe": -0.7247, "sortino": -0.7085, "calmar": -0.7952, "mdd": 0.1158}),
        },
        "POC-02 DSR": {
            "2023": _norm({"return": 0.0110, "sharpe": 0.2854, "sortino": 0.2836, "calmar": 0.2183, "mdd": 0.0514}),
            "2024": _norm({"return": 0.0145, "sharpe": 0.7625, "sortino": 0.7676, "calmar": 0.9245, "mdd": 0.0158}),
        },
        "POC-03 DSR+GAAH": {
            "2023": _norm({"return": 0.1381, "sharpe": 2.5492, "sortino": 2.7002, "calmar": 5.7933, "mdd": 0.0239}),
            "2024": _norm({"return": -0.0179, "sharpe": -0.5572, "sortino": -0.5599, "calmar": -0.5352, "mdd": 0.0335}),
        },
        "POC-04 Composite": {
            "2023": _norm({"return": 0.0629, "sharpe": 0.4968, "sortino": 0.5064, "calmar": 0.6259, "mdd": 0.1026}),
            "2024": _norm({"return": 0.1189, "sharpe": 0.8008, "sortino": 0.8652, "calmar": 1.2400, "mdd": 0.0967}),
        },
        "POC-05 Optuna Best": {
            "2023": _norm(m2023),
            "2024": _norm(m2024),
        },
        "Fenix ref": {
            "2023": _norm({"return": 0.3246, "sharpe": 2.4380, "sortino": 2.453, "calmar": 4.86, "mdd": 0.03}),
            "2024": _norm({"return": -0.0241, "sharpe": -0.1770, "sortino": -0.15, "calmar": -0.12, "mdd": 0.03}),
        },
    }

    benchmarks = {"2023": bench_2023, "2024": bench_2024}
    avg_scores = {}

    for name, regimes in all_results.items():
        scores = []
        for regime, m in regimes.items():
            ret = m["total_return"]
            alpha = ret - benchmarks[regime]
            cs = composite_score(m["sortino"], alpha, m["calmar"], m["max_drawdown"], sharpe=m["sharpe"])
            scores.append(cs)
            ibov = benchmarks[regime]
            print(f"{name:<25} {regime:<6} {ret:>+7.2%} {ibov:>+7.2%} {alpha:>+7.2%} {m['sharpe']:>8.4f} {m['sortino']:>8.4f} {m['calmar']:>8.4f} {m['max_drawdown']:>6.2%} {cs:>10.6f}")
        avg = sum(scores) / len(scores)
        avg_scores[name] = avg
        print(f"{'':25} {'AVG':>6} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} {'':>7} {avg:>10.6f}")
        print()

    print("=" * 110)
    print("RANKING:")
    for i, (name, avg) in enumerate(sorted(avg_scores.items(), key=lambda x: -x[1]), 1):
        marker = " <-- BEST FinRL" if i == 1 and "Fenix" not in name else ""
        marker_fenix = " (reference)" if "Fenix" in name else ""
        print(f"  {i}. {name:<25} {avg:.6f}{marker}{marker_fenix}")

    # Save
    out_dir = ROOT / "results" / "poc-05"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "poc": "poc-05-optuna",
        "n_trials": args.n_trials,
        "timesteps_per_trial": args.timesteps,
        "best_params": best.params,
        "best_value": best.value,
        "metrics_2023": m2023,
        "metrics_2024": m2024,
        "avg_composite": avg_scores.get("POC-05 Optuna Best", 0),
    }
    with open(out_dir / "result_optuna.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save study
    import joblib
    joblib.dump(study, out_dir / "optuna_study.pkl")
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
