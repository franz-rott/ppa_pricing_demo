from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class TrainFeatures:
    expected_market_price_eur_mwh: float
    capture_factor: float
    mean_abs_spread_eur_mwh: float
    std_spread_eur_mwh: float
    forecast_mae_mwh_per_mw: float
    forecast_rmse_mwh_per_mw: float
    error_abs_per_mwh: float  # E[|error|]/E[pv]


def compute_train_features(train: pd.DataFrame) -> TrainFeatures:
    price = train["price_da"]
    pv = train["pv_mwh_per_mw"]
    pv_fore = train["pv_forecast_mwh_per_mw"]
    err = train["pv_error_mwh_per_mw"]
    spreads = train["da_id_spread"]

    expected_market_price = float(price.mean())
    capture_price = float((price * pv).sum() / (pv.sum() + 1e-9))
    capture_factor = float(capture_price / (expected_market_price + 1e-9))

    mean_abs_spread = float(np.mean(np.abs(spreads)))
    std_spread = float(np.std(spreads, ddof=1))

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    error_abs_per_mwh = float(np.sum(np.abs(err)) / (np.sum(pv) + 1e-9))

    return TrainFeatures(
        expected_market_price_eur_mwh=expected_market_price,
        capture_factor=capture_factor,
        mean_abs_spread_eur_mwh=mean_abs_spread,
        std_spread_eur_mwh=std_spread,
        forecast_mae_mwh_per_mw=mae,
        forecast_rmse_mwh_per_mw=rmse,
        error_abs_per_mwh=error_abs_per_mwh,
    )

