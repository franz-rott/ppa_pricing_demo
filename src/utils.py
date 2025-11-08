from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def hourly_index(start: str, end: str, tz: str) -> pd.DatetimeIndex:
    # Closed interval inclusive of end hour
    idx = pd.date_range(start=start, end=end, freq="H", tz=tz)
    return idx


def split_train_test(df: pd.DataFrame, train_end: str, test_start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[:train_end].copy()
    test = df.loc[test_start:].copy()
    return train, test


def to_month_key(ts: pd.DatetimeIndex) -> pd.Index:
    return ts.to_period("M").astype(str)


def daily_percentiles(series: pd.Series) -> pd.DataFrame:
    # Returns per-day p25 and p75 for the given hourly series
    df = series.to_frame("x").copy()
    df["date"] = df.index.tz_convert(None).date
    agg = df.groupby("date")["x"].quantile([0.25, 0.75]).unstack(level=1).rename(columns={0.25: "p25", 0.75: "p75"})
    return agg


@dataclass
class DataPaths:
    data_dir: str
    fig_dir: str

