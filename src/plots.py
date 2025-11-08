from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "figure.figsize": (10, 5),
    })


def plot_decomposition(components: Dict[str, float], fig_path: Optional[str] = None):
    _style()
    labels = list(components.keys())
    values = np.array(list(components.values()), dtype=float)
    cumulative = np.cumsum(values)
    starts = np.concatenate(([0.0], cumulative[:-1]))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    running_total = 0.0
    connectors_x = []
    connectors_y = []

    for idx, (label, value, start) in enumerate(zip(labels, values, starts)):
        color = "#1f77b4" if value >= 0 else "#d62728"
        bar = ax.bar(idx, value, bottom=start, color=color, edgecolor="black", width=0.6)
        ax.text(
            idx,
            start + value / 2,
            f"{value:+.1f}",
            ha="center",
            va="center",
            color="white" if abs(value) > 0.5 else color,
            fontsize=10,
        )

        running_total = start + value
        connectors_x.append(idx + 0.3)
        connectors_y.append(running_total)

    # Total bar (fix price)
    total_value = cumulative[-1]
    total_idx = len(values)
    ax.bar(total_idx, total_value, bottom=0.0, color="#2ca02c", edgecolor="black", width=0.6)
    ax.text(
        total_idx,
        total_value / 2,
        f"{total_value:.1f}",
        ha="center",
        va="center",
        color="white" if abs(total_value) > 0.5 else "#2ca02c",
        fontsize=10,
    )

    # Draw connectors
    for idx in range(len(connectors_x) - 1):
        ax.plot(
            [connectors_x[idx], connectors_x[idx + 1] - 0.3],
            [connectors_y[idx], connectors_y[idx]],
            color="#555555",
            linewidth=1.0,
        )

    ax.plot(
        [connectors_x[-1], total_idx - 0.3],
        [connectors_y[-1], total_value],
        color="#555555",
        linewidth=1.0,
    )

    ax.set_xticks(range(len(labels) + 1))
    ax.set_xticklabels(labels + ["Fix price"], rotation=30, ha="right")
    ax.set_ylabel("€/MWh")
    ax.set_title("Fix price decomposition (waterfall)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.75, len(labels) + 0.75)
    plt.tight_layout()
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=150)
    return fig, ax


def plot_timeseries_price_pv(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None, fig_path: Optional[str] = None):
    _style()
    sel = df.loc[start:end] if start or end else df
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(sel.index, sel["price_da"], color="#1f77b4", label="DA Price (€/MWh)")
    ax2.plot(sel.index, sel["pv_mwh_per_mw"], color="#ff7f0e", label="PV (MWh/MW)")
    ax1.set_ylabel("€/MWh")
    ax2.set_ylabel("MWh per MW")
    ax1.set_title("DA Price and PV Output")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=150)
    return fig, ax1


def plot_soc(soc: pd.Series, fig_path: Optional[str] = None):
    _style()
    fig, ax = plt.subplots()
    ax.plot(soc.index, soc.values, color="#9467bd")
    ax.set_title("Battery State of Charge")
    ax.set_ylabel("MWh")
    ax.set_xlabel("")
    plt.tight_layout()
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=150)
    return fig, ax


def plot_cum_pnl(pnl_a: pd.Series, pnl_b: Optional[pd.Series] = None, labels=("No battery", "With battery"), fig_path: Optional[str] = None):
    _style()
    fig, ax = plt.subplots()
    ax.plot(pnl_a.index, pnl_a.cumsum(), label=labels[0], color="#1f77b4")
    if pnl_b is not None:
        ax.plot(pnl_b.index, pnl_b.cumsum(), label=labels[1], color="#2ca02c")
    ax.set_title("Cumulative PnL")
    ax.set_ylabel("EUR")
    ax.legend(loc="best")
    plt.tight_layout()
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=150)
    return fig, ax


def plot_pnl_distribution(pnl_hourly: pd.Series, fig_path: Optional[str] = None):
    _style()
    fig, ax = plt.subplots()
    sns.histplot(pnl_hourly.values, bins=50, kde=True, ax=ax, color="#1f77b4")
    ax.set_title("Hourly PnL Distribution")
    ax.set_xlabel("EUR")
    plt.tight_layout()
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path, dpi=150)
    return fig, ax
