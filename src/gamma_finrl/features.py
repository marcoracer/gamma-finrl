"""GAAH temporal markings — ported from gamma-fenix adapters/gaah/adapter.py.

Computes 6 temporal markings per asset from GAAH signal.
Validated in gamma-fenix POC-014 (orthogonal to momentum) and POC-015 (+Sharpe, -95% variance).
"""

from __future__ import annotations

import numpy as np


class GAAHAdapter:
    """GAAH temporal markings with sklearn-style fit/predict.

    Args:
        L: window size for signal generation.
        K: threshold multiplier for signal generation.
        method: window function ('triangle', 'uniform', 'poisson', 'logistic').
    """

    def __init__(self, L: int = 5, K: float = 0.5, method: str = "triangle") -> None:  # noqa: N803
        self.L = L
        self.K = K
        self.method = method

    def _make_window(self) -> np.ndarray:
        L = self.L  # noqa: N806
        if self.method == "triangle":
            return np.linspace(0, 2 / L, L)
        if self.method == "uniform":
            return np.ones(L) / L
        msg = f"Unknown method: {method}"
        raise ValueError(msg)

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Generate GAAH signal from prices."""
        window = self._make_window()
        n = len(prices)
        valid = n - self.L
        if valid <= 0:
            return np.zeros(n)
        ref = np.array(
            [np.dot(prices[k : k + self.L], window) / prices[k] - 1 for k in range(valid)],
        )
        std = ref.std() if len(ref) > 0 else 1.0
        sig = np.sign(ref) * (np.abs(ref) > self.K * std)
        full = np.zeros(n)
        full[: len(sig)] = sig
        return full

    @staticmethod
    def extract_markings(signal: np.ndarray) -> np.ndarray:
        """Extract 6 causal temporal markings from GAAH signal.

        Returns array of shape (T, 6):
        [distance_to_last_buy, distance_to_last_sell, last_signal_type,
         signal_density_10, n_buys_last_30, n_sells_last_30]
        """
        n = len(signal)
        features = np.zeros((n, 6), dtype=np.float32)

        last_buy = -999
        last_sell = -999
        last_sig = 0

        for t in range(n):
            s = signal[t]
            if s == 1:
                last_buy = t
                last_sig = 1
            elif s == -1:
                last_sell = t
                last_sig = -1

            features[t, 0] = (t - last_buy) if last_buy >= 0 else 252.0
            features[t, 1] = (t - last_sell) if last_sell >= 0 else 252.0
            features[t, 2] = last_sig
            features[t, 3] = np.count_nonzero(signal[max(0, t - 10) : t + 1]) / min(11, t + 1)
            window = signal[max(0, t - 30) : t + 1]
            features[t, 4] = np.count_nonzero(window == 1)
            features[t, 5] = np.count_nonzero(window == -1)

        return features

    def compute_features(self, prices: np.ndarray) -> np.ndarray:
        """Full pipeline: predict signal, extract markings. Returns (T, 6)."""
        signal = self.predict(prices)
        return self.extract_markings(signal)


def add_gaah_to_dataframe(
    df: "pd.DataFrame",
    tickers: list[str],
    L: int = 5,
    K: float = 0.5,
    method: str = "triangle",
) -> "pd.DataFrame":
    """Add GAAH temporal markings as columns to a FinRL-format DataFrame.

    Adds 6 columns per ticker: gaah_dist_buy, gaah_dist_sell, gaah_last_sig,
    gaah_density, gaah_n_buys, gaah_n_sells.

    Args:
        df: DataFrame with columns [date, tic, close, ...].
        tickers: list of ticker symbols.
        L: GAAH window size.
        K: GAAH threshold.
        method: window function.
    """
    import pandas as pd

    adapter = GAAHAdapter(L=L, K=K, method=method)

    all_dfs = []
    for tic in tickers:
        mask = df["tic"] == tic
        tic_df = df[mask].sort_values("date").copy()
        prices = tic_df["close"].values.astype(float)

        features = adapter.compute_features(prices)

        tic_df["gaah_dist_buy"] = features[:, 0]
        tic_df["gaah_dist_sell"] = features[:, 1]
        tic_df["gaah_last_sig"] = features[:, 2]
        tic_df["gaah_density"] = features[:, 3]
        tic_df["gaah_n_buys"] = features[:, 4]
        tic_df["gaah_n_sells"] = features[:, 5]

        all_dfs.append(tic_df)

    return pd.concat(all_dfs).sort_values(["date", "tic"]).reset_index(drop=True)


GAAH_INDICATORS = [
    "gaah_dist_buy",
    "gaah_dist_sell",
    "gaah_last_sig",
    "gaah_density",
    "gaah_n_buys",
    "gaah_n_sells",
]
