"""Custom StockTradingEnv with DSR reward — ported from gamma-fenix.

Subclasses FinRL's StockTradingEnv to replace the default reward
(delta_total_asset) with DSR (Differential Sharpe Ratio) + cost + concentration.
"""

from __future__ import annotations

import numpy as np

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from gamma_finrl.reward import DSRAState, hhi_concentration_penalty


class DSRStockTradingEnv(StockTradingEnv):
    """StockTradingEnv with DSR reward instead of delta_total_asset.

    Reward = DSR - cost_weight * cost_penalty - concentration_weight * HHI
    """

    def __init__(
        self,
        *args,
        cost_weight: float = 1.0,
        dsr_decay: float = 0.99,
        concentration_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cost_weight = cost_weight
        self.concentration_weight = concentration_weight
        self._dsr_state = DSRAState(decay=dsr_decay)

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._dsr_state.reset()
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

        # DSR
        dsr = self._dsr_state.update(portfolio_return)

        # Cost penalty
        cost_delta = info.get("cost_delta", 0.0)
        cost_penalty = cost_delta / max(new_nav, 1e-10)

        # Concentration penalty (HHI)
        conc_penalty = 0.0
        if self.concentration_weight > 0:
            prices = np.array(self.state[1 : 1 + self.stock_dim])
            shares = np.array(self.state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
            holdings_value = prices * shares
            total_holdings = holdings_value.sum()
            cash = self.state[0]
            total_assets = cash + total_holdings

            if total_assets > 0:
                weights = np.concatenate([
                    holdings_value / total_assets,
                    [cash / total_assets],
                ])
                conc_penalty = hhi_concentration_penalty(weights, self.stock_dim + 1)

        reward = dsr - self.cost_weight * cost_penalty - self.concentration_weight * conc_penalty
        reward = reward * self.reward_scaling

        return state, reward, terminated, truncated, info
