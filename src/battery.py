from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .utils import daily_percentiles


@dataclass
class BatteryResult:
    soc_mwh: pd.Series
    charge_mwh: pd.Series
    discharge_mwh: pd.Series
    arbitrage_revenue_eur: float
    absorbed_error_mwh: pd.Series
    absorption_ratio: float  # fraction of |error| absorbed


def simulate_battery(
    price_da: pd.Series,
    pv_mwh: pd.Series,
    forecast_error_mwh: pd.Series,
    power_mw: float,
    energy_mwh: float,
    eta_roundtrip: float,
    p_low: float = 0.25,
    p_high: float = 0.75,
) -> BatteryResult:
    assert 0.5 <= eta_roundtrip <= 1.0
    eff = float(np.sqrt(eta_roundtrip))  # split equally into charge/discharge
    # Per-day thresholds
    perc = daily_percentiles(price_da)
    # Map thresholds back hourly
    dates = price_da.index.tz_convert(None).date
    p25_map = perc.loc[pd.Index(dates), "p25"].values
    p75_map = perc.loc[pd.Index(dates), "p75"].values
    # Allow slider override via p_low/p_high
    # Compute quantiles per day according to slider
    # Replace with weighted interpolation between min and max to simulate user-defined percentiles
    # Here we simply scale between daily min and max via percentiles
    daily_min = price_da.groupby(pd.Grouper(freq="D")).transform("min").values
    daily_max = price_da.groupby(pd.Grouper(freq="D")).transform("max").values
    p_low_vals = daily_min + (daily_max - daily_min) * p_low
    p_high_vals = daily_min + (daily_max - daily_min) * p_high

    idx = price_da.index
    n = len(idx)
    soc = np.zeros(n)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    absorbed = np.zeros(n)
    revenue = 0.0

    # Scale per-hour energy limits
    p_limit = power_mw  # MW == MWh per hour
    e_limit = energy_mwh

    for t in range(n):
        price = price_da.iat[t]
        prev_soc = soc[t - 1] if t > 0 else 0.0
        # First: corrective action to absorb forecast error
        err = forecast_error_mwh.iat[t]
        if err > 0 and prev_soc > 0:  # shortfall -> discharge helps
            e_dis = min(p_limit, prev_soc, err / max(eff, 1e-9))
            if e_dis > 0:
                discharge[t] += e_dis
                prev_soc -= e_dis
                revenue += e_dis * eff * price
                absorbed[t] += min(err, e_dis * eff)
        elif err < 0 and prev_soc < e_limit:  # surplus -> charge helps
            avail_space = e_limit - prev_soc
            e_ch = min(p_limit, avail_space, -err)
            if e_ch > 0:
                charge[t] += e_ch
                prev_soc += e_ch * eff
                revenue -= e_ch * price
                absorbed[t] += min(-err, e_ch * eff)

        # Second: arbitrage based on thresholds with remaining capacity/power
        want_charge = price <= p_low_vals[t]
        want_discharge = price >= p_high_vals[t]
        # Arbitrage charge
        if want_charge and prev_soc < e_limit:
            avail_space = e_limit - prev_soc
            e = min(p_limit - charge[t], max(avail_space, 0.0))
            if e > 0:
                charge[t] += e
                prev_soc += e * eff
                revenue -= e * price
        # Arbitrage discharge
        if want_discharge and prev_soc > 0:
            avail_energy = prev_soc
            e = min(p_limit - discharge[t], max(avail_energy, 0.0))
            if e > 0:
                discharge[t] += e
                prev_soc -= e
                revenue += e * eff * price

        soc[t] = prev_soc

    absorbed_total = float(absorbed.sum())
    err_total = float(np.sum(np.abs(forecast_error_mwh.values))) + 1e-9
    ratio = float(absorbed_total / err_total)

    return BatteryResult(
        soc_mwh=pd.Series(soc, index=idx, name="soc_mwh"),
        charge_mwh=pd.Series(charge, index=idx, name="charge_mwh"),
        discharge_mwh=pd.Series(discharge, index=idx, name="discharge_mwh"),
        arbitrage_revenue_eur=float(revenue),
        absorbed_error_mwh=pd.Series(absorbed, index=idx, name="absorbed_error_mwh"),
        absorption_ratio=ratio,
    )
