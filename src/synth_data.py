from __future__ import annotations

"""
Synthetic hourly dataset for DA prices, PV production, DA–ID spreads, and forecast errors.

Goals
-----
- Training window (2022–2025) delivers capture factor ~0.9 and realistic intra-day/seasonal structure.
- Forward window (2026–2029) softens price level slightly but introduces richer noise so PnL paths are jagged yet upward.
- Preserves negative PV–price correlation and heavy-tailed spreads.
"""

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Config
from .utils import ensure_dir, hourly_index


@dataclass
class SyntheticData:
    df: pd.DataFrame


def _solar_capacity_factor(index: pd.DatetimeIndex, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)

    doy = index.dayofyear.values
    hour = index.hour.values

    day_length = 12 + 4.5 * np.sin(2 * np.pi * (doy - 80) / 365)
    solar_noon = 13
    shape_width = 3.8
    base = np.exp(-0.5 * ((hour - solar_noon) / shape_width) ** 2)
    seasonal = 0.55 + 0.38 * np.sin(2 * np.pi * (doy - 172) / 365)
    raw = base * seasonal

    sunrise = solar_noon - day_length / 2
    sunset = solar_noon + day_length / 2
    daylight = (hour >= sunrise) & (hour <= sunset)
    raw *= daylight.astype(float)

    dates = index.tz_convert(None).date
    uniq_dates = np.unique(dates)
    daily_weather = dict(zip(uniq_dates, rng.normal(0.0, 0.22, len(uniq_dates))))
    daily_factor = np.array([1.0 + daily_weather[d] for d in dates])

    intra_hour_noise = rng.lognormal(mean=-0.2, sigma=0.55, size=len(index))
    cf = raw * daily_factor * intra_hour_noise

    cloud_event_mask = rng.random(len(index)) < 0.02
    cf = np.where(cloud_event_mask, cf * rng.uniform(0.25, 0.55, size=len(index)), cf)
    cf = np.clip(cf, 0.0, None)

    if cf.max() > 0:
        cf = cf / cf.max() * 0.94

    return pd.Series(cf, index=index, name="pv_cf")


def _baseline_price(index: pd.DatetimeIndex, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed + 7)

    doy = index.dayofyear.values
    hour = index.hour.values
    dow = index.dayofweek.values

    seasonal = 60 + 16 * np.cos(2 * np.pi * (doy - 20) / 365)
    intraday = (
        12 * np.sin(2 * np.pi * (hour - 7) / 24)  # morning ramp
        + 14 * np.sin(2 * np.pi * (hour - 18) / 24)  # evening peak
        + 5 * np.sin(2 * np.pi * (hour - 13) / 12)  # midday dip
    )
    weekend_discount = np.isin(dow, [5, 6]).astype(float) * -4

    # AR(1) hourly noise
    eps = rng.normal(0.0, 6.0, size=len(index))
    for t in range(1, len(eps)):
        eps[t] = 0.65 * eps[t - 1] + eps[t]

    price = pd.Series(seasonal + intraday + weekend_discount + eps, index=index)

    dates = index.tz_convert(None).date
    uniq_dates = np.unique(dates)
    daily_increments = rng.normal(0.0, 1.2, len(uniq_dates))
    daily_regime = np.cumsum(daily_increments)
    daily_regime -= daily_regime.mean()
    daily_regime += 2.5 * np.sin(2 * np.pi * (np.arange(len(uniq_dates)) / 45.0))
    regime_map = dict(zip(uniq_dates, daily_regime))
    price += np.array([regime_map[d] for d in dates])

    daily_shocks = dict(zip(uniq_dates, rng.normal(0.0, 4.0, len(uniq_dates))))
    price += np.array([daily_shocks[d] for d in dates])

    event_mask = rng.random(len(index)) < 0.03
    event_size = rng.normal(0.0, 20.0, size=len(index))
    price += event_mask * event_size

    price += rng.normal(0.0, 4.5, size=len(index))
    price += 18.0

    # Softening forward expectations from 2026 onwards (downward shift and drift)
    forward_mask = index.year >= 2026
    price.loc[forward_mask] += -30.0
    if forward_mask.any():
        days_forward = (index[forward_mask] - index[forward_mask][0]) / np.timedelta64(1, "D")
        months_forward = days_forward / 30.0
        drift = -0.2 * months_forward
        price.loc[forward_mask] += drift
        price.loc[forward_mask] += rng.normal(0.0, 6.0, size=forward_mask.sum())

    price = price.clip(lower=-60.0)
    return price.rename("price_da")


def _apply_pv_price_correlation(
    price: pd.Series, pv_cf: pd.Series, target_capture: float, max_iter: int = 12
) -> Tuple[pd.Series, float]:
    pv_norm = (pv_cf - pv_cf.min()) / (pv_cf.max() - pv_cf.min() + 1e-9)
    alpha = 6.0
    p = price.copy()
    train_mask = p.index.year <= 2025
    for _ in range(max_iter):
        p_adj = p - alpha * pv_norm
        cap_price = (p_adj[train_mask] * pv_cf[train_mask]).sum() / (pv_cf[train_mask].sum() + 1e-9)
        mean_price = p_adj[train_mask].mean()
        capture = float(cap_price / (mean_price + 1e-9))
        err = capture - target_capture
        if abs(err) < 0.01:
            return p_adj.rename("price_da"), alpha
        alpha *= 1 - np.clip(err, -0.25, 0.25) * 2.0
        alpha = float(np.clip(alpha, 0.0, 40.0))
        p = p_adj
    return p.rename("price_da"), alpha


def _spreads(index: pd.DatetimeIndex, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed + 17)
    hour = index.hour.values
    vol = 3.5 + 2.5 * np.exp(-0.5 * ((hour - 12) / 3.0) ** 2) + 1.8 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    base = rng.standard_t(df=3, size=len(index))
    spread = base * vol
    blow_mask = rng.random(len(index)) < 0.02
    blow_size = rng.normal(0.0, 12.0, size=len(index))
    spread += blow_mask * blow_size
    dow = index.dayofweek.values
    spread += np.where(np.isin(dow, [0, 4]), rng.normal(2.5, 2.0, size=len(spread)), 0.0)
    return pd.Series(spread, index=index, name="da_id_spread")


def _pv_forecast(pv_cf: pd.Series, seed: int) -> Tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed + 23)
    hour = pv_cf.index.hour.values
    frac_sigma = 0.14
    shape = 0.5 + 0.65 * np.exp(-0.5 * ((hour - 12) / 4.0) ** 2)
    rel_err = rng.normal(0.0, frac_sigma * shape, size=len(pv_cf))
    err = rel_err * np.maximum(pv_cf.values, 0.05)
    bias_mask = rng.random(len(pv_cf)) < 0.03
    err += bias_mask * rng.normal(-0.1, 0.08, size=len(pv_cf))
    forecast = np.clip(pv_cf.values + err, 0.0, None)
    return (
        pd.Series(forecast, index=pv_cf.index, name="pv_forecast_mwh_per_mw"),
        pd.Series(err, index=pv_cf.index, name="pv_error_mwh_per_mw"),
    )


def generate_or_load(cfg: Config) -> SyntheticData:
    ensure_dir(cfg.DATA_DIR)
    path = f"{cfg.DATA_DIR}/synthetic_data.csv"
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path, parse_dates=["ts"], index_col="ts")
            dt_idx = pd.to_datetime(df.index, utc=False, errors="coerce")
            if getattr(dt_idx, "tz", None) is None:
                dt_idx = dt_idx.tz_localize(cfg.TIMEZONE)
            else:
                dt_idx = dt_idx.tz_convert(cfg.TIMEZONE)
            df.index = dt_idx
            df = df.sort_index()
            if df.index.hasnans or not df.index.is_monotonic_increasing:
                raise ValueError("Invalid cached data")
            return SyntheticData(df=df)
        except Exception:
            pass  # regenerate

    index = hourly_index(cfg.TRAIN_START, cfg.TEST_END, cfg.TIMEZONE)
    pv_cf = _solar_capacity_factor(index, cfg.RANDOM_SEED)
    price_raw = _baseline_price(index, cfg.RANDOM_SEED)
    price_da, _ = _apply_pv_price_correlation(price_raw, pv_cf, cfg.TARGET_CAPTURE)
    spread = _spreads(index, cfg.RANDOM_SEED)
    pv_fore, pv_err = _pv_forecast(pv_cf, cfg.RANDOM_SEED)

    df = pd.DataFrame(
        {
            "price_da": price_da,
            "pv_mwh_per_mw": pv_cf,
            "da_id_spread": spread,
            "pv_forecast_mwh_per_mw": pv_fore,
            "pv_error_mwh_per_mw": pv_err,
        }
    )
    df["split"] = np.where(
        df.index <= pd.Timestamp(cfg.TRAIN_END).tz_localize(cfg.TIMEZONE),
        "train",
        "test",
    )

    df_out = df.copy()
    df_out.index.name = "ts"
    df_out.to_csv(path)
    return SyntheticData(df=df)
