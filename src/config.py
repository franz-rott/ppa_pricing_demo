"""
Global configuration parameters for the synthetic Solar PPA demo.

These serve as defaults. The notebook UI can override these at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Plant and battery sizing
    PLANT_POWER_MW: float = 20.0
    BATTERY_POWER_MW: float = 10.0
    BATTERY_CAP_MWH: float = 20.0

    # Battery round-trip efficiency split as charge/discharge
    EFF_CH: float = 0.90
    EFF_DIS: float = 0.90

    # Time bounds (inclusive)
    TRAIN_START: str = "2022-01-01"
    TRAIN_END: str = "2025-12-31"
    TEST_START: str = "2026-01-01"
    TEST_END: str = "2029-12-31"

    # Timezone and resolution
    TIMEZONE: str = "Europe/Berlin"
    FREQ: str = "H"  # hourly

    # Pricing knobs (€/MWh)
    COSTS_EUR_MWH: float = 0.5
    MARGIN_EUR_MWH: float = 0.5
    OTHER_RISKS_EUR_MWH: float = 0.0

    # Training calibration targets
    TARGET_CAPTURE: float = 0.90

    # Risk weights (lambda multipliers)
    LAMBDA_MARKET: float = 1.0
    LAMBDA_PROFILE: float = 1.0
    LAMBDA_VOL: float = 1.0

    # Randomness
    RANDOM_SEED: int = 42

    # Data/figure paths (relative to repo root)
    DATA_DIR: str = "ppa_demo/data"
    FIG_DIR: str = "ppa_demo/figures"

