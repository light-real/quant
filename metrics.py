"""Daily research statistics, including the first return after the initial NAV."""
from __future__ import annotations

from numbers import Integral, Number

import numpy as np
import pandas as pd

ANN = np.sqrt(365)


def _utc_timestamp(value) -> pd.Timestamp:
    """Interpret naive date bounds as UTC and convert aware bounds to UTC."""
    if isinstance(value, Number):
        raise ValueError("Date bounds must be dates, not numeric timestamps")
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("Date bounds cannot be NaT")
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid date bound: {value!r}") from exc


def _daily_returns(equity: pd.Series) -> pd.Series:
    """Validate the entire input before deriving returns; never fill missing NAV."""
    if not isinstance(equity, pd.Series) or equity.empty:
        raise ValueError("Equity must be a nonempty Series with an initial NAV snapshot")
    index = equity.index
    if not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC":
        raise ValueError("Equity must have a UTC DatetimeIndex")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Equity dates must be finite, unique, and increasing")
    if not index.equals(index.normalize()) or not ((index[1:] - index[:-1]) == pd.Timedelta(days=1)).all():
        raise ValueError("Equity must contain consecutive daily midnight UTC snapshots")
    if (not pd.api.types.is_numeric_dtype(equity.dtype)
            or pd.api.types.is_complex_dtype(equity.dtype)
            or pd.api.types.is_bool_dtype(equity.dtype)):
        raise ValueError("Equity NAV must be real numeric values")
    values = equity.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or not (values > 0).all():
        raise ValueError("Equity NAV must be finite and strictly positive")
    return equity.astype(float).pct_change(fill_method=None)


def perf_stats(equity: pd.Series, start=None, end=None) -> dict:
    """Select inclusive return dates, retaining the prior NAV for the first return."""
    returns = _daily_returns(equity).iloc[1:]
    start = _utc_timestamp(start) if start is not None else None
    end = _utc_timestamp(end) if end is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("start must not be after end")
    if start is not None:
        returns = returns.loc[returns.index >= start]
    if end is not None:
        returns = returns.loc[returns.index <= end]
    n_days = len(returns)
    if not n_days:
        return {"n_days": 0, "total_return": None, "cagr": None,
                "vol_annual": None, "sharpe": None, "max_dd": None, "calmar": None}

    growth = float((1 + returns).prod())
    cagr = growth ** (365 / n_days) - 1
    sd = float(returns.std(ddof=1)) if n_days > 1 else None
    curve = np.concatenate(([1.0], (1 + returns).cumprod().to_numpy()))
    max_dd = float((1 - curve / np.maximum.accumulate(curve)).max())
    return {
        "n_days": n_days,
        "total_return": growth - 1,
        "cagr": cagr,
        "vol_annual": sd * ANN if sd is not None else None,
        "sharpe": float(returns.mean() / sd * ANN) if sd is not None and sd > 0 else None,
        "max_dd": max_dd,
        "calmar": cagr / max_dd if max_dd > 0 else None,
    }


def annual_returns(equity: pd.Series) -> pd.DataFrame:
    """Compound full-history daily returns by their year, including January 1."""
    returns = _daily_returns(equity).iloc[1:]
    rows = [
        {"year": int(year), "return": float((1 + group).prod() - 1), "n_days": len(group)}
        for year, group in returns.groupby(returns.index.year)
    ]
    return pd.DataFrame(rows, columns=["year", "return", "n_days"]).astype(
        {"year": "int64", "return": "float64", "n_days": "int64"}
    )


def rolling_stats(equity: pd.Series, window=365) -> pd.DataFrame:
    """Trailing full-window daily-return statistics; unobserved windows are NaN."""
    if isinstance(window, bool) or not isinstance(window, Integral) or window < 2:
        raise ValueError("window must be an integer >= 2")
    returns = _daily_returns(equity)
    rolling = returns.rolling(window=int(window), min_periods=int(window))
    sd = rolling.std(ddof=1)
    return pd.DataFrame({
        "rolling_return": rolling.apply(lambda values: np.prod(1 + values) - 1, raw=True),
        "rolling_vol": sd * ANN,
        "rolling_sharpe": rolling.mean() / sd.where(sd > 0) * ANN,
    }, index=equity.index)
