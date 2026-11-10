"""File-only research charts for observed NAV and realized EOD exposure."""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

from metrics import _utc_timestamp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _finite_values(data: pd.Series | pd.DataFrame) -> np.ndarray:
    if not isinstance(data, (pd.Series, pd.DataFrame)) or data.empty:
        raise ValueError("Chart data must be nonempty")
    try:
        values = data.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("Chart data must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Chart data must be finite, with no missing observations")
    return values


def equity_chart(series_map: dict[str, pd.Series], path, title: str, vline=None):
    """Plot initial-1 NAV and signed drawdown on aligned axes."""
    if not series_map:
        raise ValueError("At least one NAV series is required")
    normalized = {}
    for label, equity in series_map.items():
        values = _finite_values(equity)
        if values[0] <= 0:
            raise ValueError("Initial NAV must be positive for normalization")
        normalized[label] = pd.Series(values / values[0], index=equity.index)
    boundary = _utc_timestamp(vline) if vline is not None else None

    fig, (nav_ax, dd_ax) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    try:
        for label, nav in normalized.items():
            line, = nav_ax.plot(nav.index, nav.to_numpy(), label=label, linewidth=1.2)
            drawdown = nav / nav.cummax() - 1
            dd_ax.plot(nav.index, drawdown.to_numpy(), label=label,
                       color=line.get_color(), linewidth=1.2)
        if all((nav > 0).all() for nav in normalized.values()):
            nav_ax.set_yscale("log")
        if boundary is not None:
            for axis in (nav_ax, dd_ax):
                axis.axvline(boundary, color="gray", linestyle="--", linewidth=0.9,
                             label="historical validation (previously inspected)")
        nav_ax.set_title(title)
        nav_ax.set_ylabel("Normalized NAV (initial = 1)")
        nav_ax.legend(fontsize=8, ncol=2 if len(normalized) > 3 else 1)
        dd_ax.set_ylabel("Drawdown")
        dd_ax.yaxis.set_major_formatter(PercentFormatter(1))
        dd_ax.set_xlabel("Date (UTC)")
        for axis in (nav_ax, dd_ax):
            axis.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, format="png", dpi=130)
    finally:
        plt.close(fig)


def exposure_chart(weights: pd.DataFrame, path, title: str):
    """Plot long, absolute short, and gross exposure from realized EOD weights."""
    if not isinstance(weights, pd.DataFrame):
        raise ValueError("weights must be a DataFrame of realized EOD weights")
    values = _finite_values(weights)
    long = np.clip(values, 0, None).sum(axis=1)
    short = -np.clip(values, None, 0).sum(axis=1)
    gross = np.abs(values).sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    try:
        for label, exposure in (("Long", long), ("Short", short), ("Gross", gross)):
            ax.plot(weights.index, exposure, label=label, linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Realized EOD exposure (NAV fraction)")
        ax.set_xlabel("Date (UTC)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, format="png", dpi=130)
    finally:
        plt.close(fig)
