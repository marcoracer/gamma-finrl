"""Reward functions — ported from gamma-fenix domain.

DSR (Differential Sharpe Ratio) + cost penalty + concentration penalty.
Based on Moody & Saffell (2001) and gamma-fenix ADR-011/ADR-016.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSRAState:
    """State for incremental Differential Sharpe Ratio calculation.

    Based on Moody & Saffell (2001): "Reinforcement Learning for Trading".
    Uses exponential moving averages to update Sharpe estimate online.
    """

    mean_return: float = 0.0
    mean_sq_return: float = 0.0
    decay: float = 0.99  # exponential moving average decay

    def update(self, portfolio_return: float) -> float:
        """Update DSR state with new return and return the DSR value."""
        delta_mean = portfolio_return - self.mean_return
        delta_var = portfolio_return**2 - self.mean_sq_return

        self.mean_return += self.decay * delta_mean
        self.mean_sq_return += self.decay * delta_var

        variance = self.mean_sq_return - self.mean_return**2
        if variance <= 1e-12:
            return 0.0

        std = variance**0.5
        dsr = (std * delta_mean - 0.5 * self.mean_return * delta_var) / (std**3)
        return dsr

    def reset(self) -> None:
        self.mean_return = 0.0
        self.mean_sq_return = 0.0


def hhi_concentration_penalty(weights: list[float] | tuple[float, ...], n_assets: int) -> float:
    """Herfindahl-Hirschman concentration penalty (ADR-016).

    HHI = sum(w_i^2) — ranges from 1/N (equal) to 1.0 (single stock).
    Penalty is zero at equal weights, maximal at single stock.
    """
    hhi = sum(w * w for w in weights)
    equal_hhi = 1.0 / n_assets
    return max(0.0, hhi - equal_hhi)
