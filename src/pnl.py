from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class PnLResult:
    hourly: pd.DataFrame
    summary: Dict[str, float]
    annual: pd.DataFrame


def pnl_no_battery(test: pd.DataFrame, fix_price_eur_mwh: float, plant_power_mw: float, err_scale: float = 1.0, spread_scale: float = 1.0) -> PnLResult:
    df = test.copy()
    pv = df["pv_mwh_per_mw"] * plant_power_mw
    price = df["price_da"]
    spreads = df["da_id_spread"] * spread_scale
    error = (df["pv_forecast_mwh_per_mw"] - df["pv_mwh_per_mw"]) * err_scale * plant_power_mw

    imbalance_cost = np.abs(error) * np.abs(spreads)
    pnl = fix_price_eur_mwh * pv - price * pv - imbalance_cost

    out = df[["price_da", "pv_mwh_per_mw", "pv_forecast_mwh_per_mw", "da_id_spread"]].copy()
    out["pv_mwh"] = pv
    out["error_mwh"] = error
    out["imbalance_cost_eur"] = imbalance_cost
    out["pnl_eur"] = pnl

    summary = _summary_from_hourly(out["pnl_eur"]) 
    annual = _annual_from_hourly(out["pnl_eur"]) 
    return PnLResult(hourly=out, summary=summary, annual=annual)


def pnl_with_battery(test: pd.DataFrame, fix_price_eur_mwh: float, plant_power_mw: float, spreads_scale: float, err_scale: float, batt_result, eta_roundtrip: float = 0.81) -> PnLResult:
    df = test.copy()
    pv = df["pv_mwh_per_mw"] * plant_power_mw
    price = df["price_da"]
    spreads = df["da_id_spread"] * spreads_scale
    error = (df["pv_forecast_mwh_per_mw"] - df["pv_mwh_per_mw"]) * err_scale * plant_power_mw

    absorbed = batt_result.absorbed_error_mwh
    corrected_error = np.maximum(np.abs(error) - absorbed, 0.0)
    imbalance_cost = corrected_error * np.abs(spreads)

    # Battery arbitrage revenue already computed as total; distribute proportionally per hour for display
    # Here, compute per-hour revenue approximation
    charge = batt_result.charge_mwh
    discharge = batt_result.discharge_mwh
    eff = float(np.sqrt(eta_roundtrip))
    batt_hourly_rev = (discharge * eff * price - charge * price).rename("batt_revenue_eur")
    total_batt_rev = batt_hourly_rev.sum()

    pnl = fix_price_eur_mwh * pv - price * pv - imbalance_cost + batt_hourly_rev

    out = df[["price_da", "pv_mwh_per_mw", "pv_forecast_mwh_per_mw", "da_id_spread"]].copy()
    out["pv_mwh"] = pv
    out["error_mwh"] = error
    out["absorbed_error_mwh"] = absorbed
    out["imbalance_cost_eur"] = imbalance_cost
    out["batt_revenue_eur"] = batt_hourly_rev
    out["pnl_eur"] = pnl

    summary = _summary_from_hourly(out["pnl_eur"]) 
    summary["battery_revenue_total_eur"] = float(batt_hourly_rev.sum())
    annual = _annual_from_hourly(out["pnl_eur"]) 
    return PnLResult(hourly=out, summary=summary, annual=annual)


def _summary_from_hourly(series: pd.Series) -> Dict[str, float]:
    arr = series.values
    return {
        "total_eur": float(arr.sum()),
        "mean_hourly_eur": float(arr.mean()),
        "std_hourly_eur": float(np.std(arr, ddof=1)),
        "p5_hourly_eur": float(np.quantile(arr, 0.05)),
    }


def _annual_from_hourly(series: pd.Series) -> pd.DataFrame:
    df = series.to_frame("pnl_eur").copy()
    df["year"] = df.index.year
    return df.groupby("year").sum()
