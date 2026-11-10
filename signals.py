"""Causal daily trend forecasts and covariance-aware portfolio target weights.

Outputs use information through the current close. Execution lag belongs to the
engine, not these functions. Passive, pure-long and dual portfolios respectively
supply all-one, positive-part and signed forecasts to the same weight function.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd

ANNUAL_DAYS = 365
VOL_FLOOR = 0.05
# Log-price subtraction can leave roundoff in otherwise deterministic returns.
ROUND_OFF = 64 * np.finfo(float).eps


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _numeric_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if any(not pd.api.types.is_numeric_dtype(dtype)
           or pd.api.types.is_bool_dtype(dtype)
           or pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"{name} must contain real numeric values")
    return frame.astype(float)


def _validated_close(close: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(close, pd.DataFrame) or len(close.columns) == 0:
        raise ValueError("close must be a DataFrame with at least one asset")
    if close.columns.has_duplicates:
        raise ValueError("close columns must be unique")
    index = close.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans:
        raise ValueError("close index must contain daily timestamps")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("close index must be unique and strictly increasing")
    utc = index.tz_convert("UTC") if index.tz is not None else index
    if (utc != utc.normalize()).any():
        raise ValueError("close timestamps must be UTC day-aligned")
    if ((utc[1:] - utc[:-1]) != pd.Timedelta(days=1)).any():
        raise ValueError("close index must be daily without gaps")
    numeric = _numeric_frame(close, "close")
    values = numeric.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("close prices must be positive and finite")
    return numeric


def trend_forecast(close: pd.DataFrame, horizons=(20, 60, 120),
                   vol_window=60) -> pd.DataFrame:
    """Average clipped log-momentum / sample-vol forecasts across all horizons.

    Every horizon and the full daily return window must be available. A
    deterministic (zero-volatility) return window contributes zero, never a
    leveraged or saturated forecast. Naive daily indices are interpreted as UTC.
    """
    close = _validated_close(close)
    vol_window = _positive_integer(vol_window, "vol_window")
    try:
        horizons = tuple(horizons)
    except TypeError as exc:
        raise ValueError("horizons must contain positive integers") from exc
    if not horizons:
        raise ValueError("horizons must not be empty")
    horizons = tuple(_positive_integer(horizon, "horizon") for horizon in horizons)

    log_close = np.log(close)
    daily_vol = log_close.diff().rolling(vol_window, min_periods=vol_window).std(ddof=1)
    daily_vol = daily_vol.mask(daily_vol <= ROUND_OFF, 0.0)
    denominator = daily_vol.replace(0.0, np.nan)
    components = []
    for horizon in horizons:
        change = log_close - log_close.shift(horizon)
        component = (change / (denominator * np.sqrt(horizon))).clip(-1.0, 1.0)
        component = component.mask(daily_vol.eq(0.0) & change.notna(), 0.0)
        components.append(component)
    # Ordinary addition preserves NaN if any component is still warming up.
    return sum(components) / len(components)


def regime_forecast(close: pd.DataFrame, ma_len=120, hysteresis=0.0) -> pd.DataFrame:
    """Binary long/flat regime from the close relative to its own moving average.

    A close above ``ma * (1 + hysteresis)`` turns the regime on, a close below
    ``ma * (1 - hysteresis)`` turns it off, and anything inside the band carries
    the previous state forward. Warmup rows, where the average is still
    incomplete, stay at zero rather than borrowing a partially formed level.
    """
    close = _validated_close(close)
    ma_len = _positive_integer(ma_len, "ma_len")
    if (isinstance(hysteresis, (bool, np.bool_)) or not isinstance(hysteresis, Real)
            or not np.isfinite(hysteresis) or not 0 <= hysteresis < 1):
        raise ValueError("hysteresis must be a finite fraction in [0, 1)")

    average = close.rolling(ma_len, min_periods=ma_len).mean()
    values = close.to_numpy()
    upper = (values > average.to_numpy() * (1 + hysteresis))
    lower = (values < average.to_numpy() * (1 - hysteresis))
    valid = average.notna().to_numpy()
    state = np.zeros(close.shape[1])
    result = np.zeros(close.shape)
    for t in range(len(close)):
        if not valid[t].all():
            continue
        state = np.where(upper[t], 1.0, np.where(lower[t], 0.0, state))
        result[t] = state
    return pd.DataFrame(result, index=close.index, columns=close.columns)


def regime_weights(close: pd.DataFrame, regime: pd.DataFrame, vol_window=60,
                   target_vol=0.60, max_gross=1.8, max_asset=0.9) -> pd.DataFrame:
    """Risk-target the assets that are currently in their long regime.

    Inverse-volatility weights are normalized over the active subset only, so a
    dormant asset does not silently absorb half of the risk budget. The result
    is risk-targeted, clipped per asset, gross-capped and finally re-checked;
    every adjustment can only reduce exposure. Empty regimes produce zero
    weights rather than a partial allocation.
    """
    close = _validated_close(close)
    vol_window = _positive_integer(vol_window, "vol_window")
    if (not isinstance(regime, pd.DataFrame) or not regime.index.equals(close.index)
            or not regime.columns.equals(close.columns)):
        raise ValueError("regime indices and columns must match close exactly")
    regime = _numeric_frame(regime, "regime")
    regime_values = regime.to_numpy()
    if np.isinf(regime_values).any() or not np.isin(regime_values[~np.isnan(regime_values)],
                                                    (0.0, 1.0)).all():
        raise ValueError("regime must contain only 0, 1 or NaN during warmup")
    for name, value in (("target_vol", target_vol), ("max_gross", max_gross),
                        ("max_asset", max_asset)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < target_vol <= 1:
        raise ValueError("target_vol must be an annual fraction in (0, 1]")

    result = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if vol_window < 2 or max_asset == 0 or max_gross == 0:
        return result
    returns = np.log(close).diff().to_numpy()
    for t in range(vol_window, len(close)):
        active = regime_values[t] > 0.5
        if not active.any():
            continue
        sample = returns[t - vol_window + 1:t + 1]
        covariance = np.atleast_2d(np.cov(sample, rowvar=False, ddof=1)) * ANNUAL_DAYS
        deterministic = np.diag(covariance) <= ROUND_OFF ** 2 * ANNUAL_DAYS
        covariance[deterministic, :] = 0.0
        covariance[:, deterministic] = 0.0
        sigma = np.maximum(np.sqrt(np.diag(covariance)), VOL_FLOOR)
        base = np.where(active, 1.0 / sigma, 0.0)
        base = base / base.sum()
        base_risk = _risk(base, covariance)
        if base_risk == 0:
            continue

        weights = base * (target_vol / base_risk)
        weights = np.clip(weights, -max_asset, max_asset)
        gross = np.abs(weights).sum()
        if gross > max_gross:
            weights *= max_gross / gross
        portfolio_risk = _risk(weights, covariance)
        if portfolio_risk == 0:
            continue
        if portfolio_risk > target_vol:
            weights *= target_vol / portfolio_risk
        result.iloc[t] = weights
    return result


def _risk(weights: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weights @ covariance @ weights)
    cancellation_bound = ROUND_OFF * float(np.abs(weights) @ np.abs(covariance) @ np.abs(weights))
    return float(np.sqrt(variance)) if variance > cancellation_bound else 0.0


def portfolio_weights(close: pd.DataFrame, forecast: pd.DataFrame, vol_window=60,
                      target_vol=0.15, max_gross=1.0, max_asset=0.6) -> pd.DataFrame:
    """Size inverse-volatility base weights, then attenuate them by forecasts.

    Sample log-return covariance is annualized by 365; inverse-vol weights use
    a 5% annual volatility floor. The unattenuated base is risk-targeted first.
    Asset/gross caps and a final covariance check can only scale exposure down.
    Missing forecast rows and zero estimated portfolio risk produce zero weights.
    target_vol is an annual fraction in (0, 1]; caps are finite and nonnegative.
    """
    close = _validated_close(close)
    vol_window = _positive_integer(vol_window, "vol_window")
    if (not isinstance(forecast, pd.DataFrame)
            or not forecast.index.equals(close.index)
            or not forecast.columns.equals(close.columns)):
        raise ValueError("forecast indices and columns must match close exactly")
    forecast = _numeric_frame(forecast, "forecast")
    forecast_values = forecast.to_numpy()
    if np.isinf(forecast_values).any() or (np.abs(forecast_values) > 1).any():
        raise ValueError("forecast must be finite in [-1, 1] or NaN during warmup")
    for name, value in (("target_vol", target_vol), ("max_gross", max_gross),
                        ("max_asset", max_asset)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < target_vol <= 1:
        raise ValueError("target_vol must be an annual fraction in (0, 1]")

    result = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if vol_window < 2 or max_asset == 0 or max_gross == 0:
        return result
    returns = np.log(close).diff().to_numpy()
    for t in range(vol_window, len(close)):
        current_forecast = forecast_values[t]
        if not np.isfinite(current_forecast).all():
            continue
        sample = returns[t - vol_window + 1:t + 1]
        covariance = np.atleast_2d(np.cov(sample, rowvar=False, ddof=1)) * ANNUAL_DAYS
        deterministic = np.diag(covariance) <= ROUND_OFF ** 2 * ANNUAL_DAYS
        covariance[deterministic, :] = 0.0
        covariance[:, deterministic] = 0.0
        sigma = np.maximum(np.sqrt(np.diag(covariance)), VOL_FLOOR)
        base = (1.0 / sigma) / len(close.columns)
        base_risk = _risk(base, covariance)
        if base_risk == 0:
            continue

        weights = base * (target_vol / base_risk) * current_forecast
        weights = np.clip(weights, -max_asset, max_asset)
        gross = np.abs(weights).sum()
        if gross > max_gross:
            weights *= max_gross / gross
        forecast_risk = _risk(weights, covariance)
        if forecast_risk == 0:
            continue
        if forecast_risk > target_vol:
            weights *= target_vol / forecast_risk
        result.iloc[t] = weights
    return result
