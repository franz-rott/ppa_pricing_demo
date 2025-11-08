from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd

from .risk_premia import (
    market_price_risk_premium,
    profile_imbalance_premium_per_mwh,
    intraday_vol_premium_per_mwh,
)


@dataclass
class PriceBreakdown:
    components: Dict[str, float]
    fix_price_eur_mwh: float


def compute_fix_price(
    train_df: pd.DataFrame,
    expected_market_price: float,
    capture_factor: float,
    lam_market: float,
    lam_profile: float,
    lam_vol: float,
    costs_eur_mwh: float,
    margin_eur_mwh: float,
    other_risks_eur_mwh: float,
    err_scale: float = 1.0,
    spread_scale: float = 1.0,
    battery_mitigation: float = 0.0,
) -> PriceBreakdown:
    """
    Compute fixed price and component breakdown (€/MWh), using training data.
    battery_mitigation: fraction [0,1] by which profile/imbalance premium is reduced due to storage.
    """
    base = expected_market_price * capture_factor

    # Risk premia per MWh
    mkt_prem = market_price_risk_premium(train_df["price_da"], horizon_months=48, lam=lam_market)

    # Compute error_abs_per_mwh directly from train
    pv = train_df["pv_mwh_per_mw"]
    err = train_df["pv_error_mwh_per_mw"]
    spreads = train_df["da_id_spread"]
    error_abs_per_mwh = float((abs(err)).sum() / (pv.sum() + 1e-9))
    mean_abs_spread = float(abs(spreads).mean())
    std_spread = float(spreads.std(ddof=1))

    prof_prem = profile_imbalance_premium_per_mwh(error_abs_per_mwh, mean_abs_spread, lam_profile, err_scale, spread_scale)
    vol_prem = intraday_vol_premium_per_mwh(error_abs_per_mwh, std_spread, lam_vol, err_scale, spread_scale)

    # Battery mitigation applied to profile premium
    prof_prem_after = prof_prem * (1.0 - max(min(battery_mitigation, 1.0), 0.0))

    components = {
        "Expected×Capture": base,
        "− Market price risk": -mkt_prem,
        "− Profile/imbalance": -prof_prem_after,
        "− Intraday volatility": -vol_prem,
        "− Costs": -costs_eur_mwh,
        "− Margin": -margin_eur_mwh,
        "− Other risks": -other_risks_eur_mwh,
    }
    fix_price = sum(components.values())
    return PriceBreakdown(components=components, fix_price_eur_mwh=float(fix_price))

