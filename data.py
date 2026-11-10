"""Load complete UTC daily bars from a local, acquisition-dated CSV snapshot."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
USECOLS = ["open_time", "close_time", *OHLCV]
DAY_MS = 86_400_000


def _utc_timestamp(value, label: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("missing timestamp")
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {label} timestamp") from exc


def load_daily(symbol: str, csv_dir="csv", as_of=None) -> pd.DataFrame:
    """Return OHLCV bars closed strictly before both as_of and snapshot creation.

    Naive cutoff dates mean UTC. Without as_of, the wall clock is used; an
    available manifest always limits it, so an unfinished snapshot bar cannot
    become complete merely because it is loaded later. Invalid source rows are
    rejected, not sorted or repaired; continuity is required only after cutoff.
    """
    if not isinstance(symbol, str) or re.fullmatch(r"[A-Za-z0-9]+", symbol) is None:
        raise ValueError("symbol must be a nonempty alphanumeric ticker")

    directory = Path(csv_dir)
    cutoff = pd.Timestamp.now(tz="UTC") if as_of is None else _utc_timestamp(as_of, "as_of cutoff")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
        except ValueError as exc:
            raise ValueError("invalid manifest.json") from exc
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json must contain an object")
        if "generated_utc" in manifest:
            generated = _utc_timestamp(manifest["generated_utc"], "manifest generated_utc")
            cutoff = min(cutoff, generated)

    raw = pd.read_csv(directory / f"{symbol}_1d.csv", usecols=USECOLS)
    if raw.empty:
        raise ValueError(f"{symbol}: empty daily data")
    try:
        raw = raw.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{symbol}: malformed numeric OHLCV or timestamp") from exc
    if any(pd.api.types.is_bool_dtype(dtype) for dtype in raw.dtypes):
        raise ValueError(f"{symbol}: malformed numeric OHLCV or timestamp")
    if not np.isfinite(raw[USECOLS].to_numpy(dtype=float)).all():
        raise ValueError(f"{symbol}: nonfinite OHLCV or timestamp")

    opens, closes = raw["open_time"], raw["close_time"]
    if opens.duplicated().any():
        raise ValueError(f"{symbol}: duplicate open_time")
    if not opens.is_monotonic_increasing:
        raise ValueError(f"{symbol}: open_time must be strictly increasing")
    if (opens % DAY_MS != 0).any():
        raise ValueError(f"{symbol}: open_time must be UTC day-aligned")
    if (closes != opens + DAY_MS - 1).any():
        raise ValueError(f"{symbol}: close_time must equal open_time + 86399999")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(opens, unit="ms", utc=True), name="open_time")
        close_index = pd.to_datetime(closes, unit="ms", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{symbol}: invalid daily timestamp") from exc

    frame = raw[OHLCV].astype(float)
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive OHLC price")
    if (frame["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume")
    if ((frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any()
            or (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any()):
        raise ValueError(f"{symbol}: invalid OHLC bounds")

    frame.index = index
    retained = frame.loc[(close_index < cutoff).to_numpy()]
    if retained.empty:
        raise ValueError(f"{symbol}: no complete retained daily bars")
    if ((retained.index[1:] - retained.index[:-1]) != pd.Timedelta(days=1)).any():
        raise ValueError(f"{symbol}: gap in retained daily series")
    return retained
