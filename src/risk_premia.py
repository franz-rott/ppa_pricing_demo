from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


def market_price_risk_premium(train_prices: pd.Series, horizon_months: int, lam: float, seed: int = 1234) -> float:
    """
    Long-term market price uncertainty premium (€/MWh).
    Approach: bootstrap monthly means from training, simulate horizon average, take mean-to-p5 downside and scale by λ.
    """
    # Monthly means in train
    monthly = train_prices.resample("M").mean().dropna().values
    rng = np.random.default_rng(seed)
    n = 5000
    sims = []
    for _ in range(n):
        draw = rng.choice(monthly, size=horizon_months, replace=True)
        sims.append(draw.mean())
    sims = np.array(sims)
    mu = float(np.mean(sims))
    p5 = float(np.quantile(sims, 0.05))
    premium = lam * max(mu - p5, 0.0)
    return float(premium)


def profile_imbalance_premium_per_mwh(error_abs_per_mwh: float, mean_abs_spread: float, lam: float, err_scale: float = 1.0, spread_scale: float = 1.0) -> float:
    """Short-term profile/imbalance premium per MWh delivered (€/MWh)."""
    return float(lam * error_abs_per_mwh * err_scale * mean_abs_spread * spread_scale)


def intraday_vol_premium_per_mwh(error_abs_per_mwh: float, std_spread: float, lam: float, err_scale: float = 1.0, spread_scale: float = 1.0) -> float:
    """Short-term intraday volatility premium per MWh delivered (€/MWh)."""
    return float(lam * error_abs_per_mwh * err_scale * std_spread * spread_scale * 0.8)

