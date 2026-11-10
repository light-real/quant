#!/usr/bin/env python3
"""Independent audit of the fetched CSVs: continuity, cross-interval agreement, sanity.

Section 1 proves each file is an unbroken series except where Binance itself has
no candles (engine downtime). Section 2 re-aggregates fine bars up and demands
that the coarse file agrees, but only for coarse buckets whose fine bars are all
present -- an outage-truncated bucket legitimately disagrees.
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

UTC = dt.timezone.utc
PERIOD = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000,
          "1d": 86_400_000, "1w": 604_800_000, "1M": 2_592_000_000,
          "1y": 31_536_000_000}
COLS = ["open_time", "open", "high", "low", "close", "volume"]
FIXED = ("5m", "15m", "1h", "4h", "1d", "1w")
# Weekly bars open Monday 00:00 UTC, not on the Unix epoch (a Thursday).
ANCHOR = {"1w": 345_600_000}


def iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M")


def load(sym, iv):
    return pd.read_csv(f"csv/{sym}_{iv}.csv", usecols=COLS)


def steps(df, iv):
    """Discontinuities between consecutive bars, as (from, to, lost_hours)."""
    ts = df["open_time"].to_numpy(dtype="int64")
    if iv not in FIXED or ts.size < 2:
        return ts, []
    p = PERIOD[iv]
    d = np.diff(ts)
    bad = np.nonzero(d != p)[0]
    found = [(ts[i], ts[i + 1], (d[i] - p) / 3_600_000) for i in bad]
    return ts, found


def main():
    print("=" * 96)
    print("1) CONTINUITY  (bars present vs bars a 24/7 series would have over the same span)")
    print("=" * 96)
    outage_table = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        for iv in ("5m", "15m", "1h", "4h", "1d", "1w", "1M", "1y"):
            df = load(sym, iv)
            ts, holes = steps(df, iv)
            span = ts[-1] - ts[0]
            expected = int(span // PERIOD[iv]) + 1 if iv in FIXED else len(ts)
            lost = round(sum(h for _a, _b, h in holes), 2)
            dupes = int(len(ts) - len(np.unique(ts)))
            verdict = "OK" if dupes == 0 else "DUPLICATE TIMESTAMPS"
            print(f"{sym:<8}{iv:>4}  bars={len(ts):>7}  expected={expected:>7} "
                  f"missing={expected - len(ts):>4}  holes={len(holes):>3} "
                  f"lost={lost:>7}h  dupes={dupes}  {verdict}  "
                  f"{iso(ts[0])} -> {iso(ts[-1])}")
            if iv == "5m":
                outage_table[sym] = holes
    print("\nBinance-side downtime windows recovered from the 5m series:")
    for sym, holes in outage_table.items():
        hours = sum(h for _a, _b, h in holes)
        print(f"  {sym}: {len(holes)} holes, {hours:.1f}h of candles absent at the source")
        for a, b, h in holes[:6]:
            print(f"     {iso(a)} -> {iso(b)}   ({h:.2f}h)")
        if len(holes) > 6:
            print(f"     ... {len(holes) - 6} more")

    print()
    print("=" * 96)
    print("2) CROSS-INTERVAL AGREEMENT  (re-aggregate finer file, compare with coarser file)")
    print("   Only buckets holding the full expected count of finer bars are judged.")
    print("=" * 96)
    failures = 0
    pairs = (("5m", "15m"), ("5m", "1h"), ("5m", "4h"), ("5m", "1d"),
             ("15m", "1h"), ("1h", "4h"), ("1h", "1d"), ("4h", "1d"), ("1d", "1w"))
    for sym in ("BTCUSDT", "ETHUSDT"):
        cache = {iv: load(sym, iv) for iv in ("5m", "15m", "1h", "4h", "1d", "1w")}
        for fine_iv, coarse_iv in pairs:
            res = compare(cache[fine_iv], fine_iv, coarse_iv, sym)
            bad = sum(res["mismatched"].values())
            failures += bad
            tag = "OK " if bad == 0 else "!! "
            print(f"{tag}{sym:<8} {fine_iv:>3}->{coarse_iv:<3} "
                  f"compared={res['full_buckets']:>6}  truncated(excluded)={res['partial']:>4}  "
                  f"mismatches={res['mismatched']}")
        failures += yearly_check(sym, cache["1d"])
        del cache

    print()
    print("=" * 96)
    print("3) HISTORICAL SANITY  (reference candles and window extremes)")
    print("=" * 96)
    for sym in ("BTCUSDT", "ETHUSDT"):
        day = load(sym, "1d")
        day["date"] = pd.to_datetime(day["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        hi, lo = day.loc[day["high"].idxmax()], day.loc[day["low"].idxmin()]
        print(f"  {sym}: max high={hi['high']:>12,.2f} on {hi['date']}   "
              f"min low={lo['low']:>10,.2f} on {lo['date']}")
        for d in ("2021-11-10", "2021-05-19", "2024-03-14", "2021-01-11"):
            row = day[day["date"] == d]
            if len(row):
                r = row.iloc[0]
                print(f"     {d}  o={r['open']:>11,.2f} h={r['high']:>11,.2f} "
                      f"l={r['low']:>11,.2f} c={r['close']:>11,.2f}")

    print(f"\ncross-interval mismatch total: {failures}")
    return 1 if failures else 0


def yearly_check(sym, daily):
    """Re-derive the yearly bars from daily closes by calendar year -- a different
    code path than the fetcher's 1M rollup, so it can catch aggregation bugs."""
    d = daily.copy()
    d["year"] = pd.to_datetime(d["open_time"], unit="ms", utc=True).dt.year
    agg = d.groupby("year").agg(open_=("open", "first"), high=("high", "max"),
                                low=("low", "min"), close=("close", "last"),
                                volume=("volume", "sum"))
    real = load(sym, "1y").copy()
    real["year"] = pd.to_datetime(real["open_time"], unit="ms", utc=True).dt.year
    r = real.set_index("year")
    common = agg.index.intersection(r.index)[:-1]  # newest year is still open
    a = agg.loc[common]
    mis = {}
    for src, dst in (("open_", "open"), ("high", "high"), ("low", "low"),
                     ("close", "close"), ("volume", "volume")):
        av = a[src].astype(float).to_numpy()
        rv = r.loc[common, dst].astype(float).to_numpy()
        rel = 1e-6 if dst == "volume" else 1e-9
        mis[dst] = int((np.abs(av - rv) > rel * (np.abs(rv) + 1)).sum())
    bad = sum(mis.values())
    print(f"{'OK ' if bad == 0 else '!! '}{sym:<8} 1d->1y  years={len(common):>3}  "
          f"mismatches={mis}  (independent of the 1M rollup)")
    return bad


def compare(fine, fine_iv, coarse_iv, sym):
    per = PERIOD[coarse_iv] // PERIOD[fine_iv]
    anchor = ANCHOR.get(coarse_iv, 0)
    g = fine.copy()
    g["bucket"] = ((g["open_time"] - anchor) // PERIOD[coarse_iv]) * PERIOD[coarse_iv] + anchor
    agg = g.groupby("bucket").agg(open_=("open", "first"), high=("high", "max"),
                                  low=("low", "min"), close=("close", "last"),
                                  volume=("volume", "sum"), n=("open", "size"))
    real = load(sym, coarse_iv).set_index("open_time")
    common = agg.index.intersection(real.index)[:-1]  # newest coarse bar is still open
    a, r = agg.loc[common], real.loc[common]
    full = (a["n"] == per).to_numpy()
    mismatched = {}
    for src, dst in (("open_", "open"), ("high", "high"), ("low", "low"),
                     ("close", "close"), ("volume", "volume")):
        av = a[src].astype(float).to_numpy()
        rv = r[dst].astype(float).to_numpy()
        rel = 1e-6 if dst == "volume" else 1e-9
        off = (np.abs(av - rv) > rel * (np.abs(rv) + 1)) & full
        mismatched[dst] = int(off.sum())
    return {"full_buckets": int(full.sum()), "partial": int((~full).sum()),
            "mismatched": mismatched}


if __name__ == "__main__":
    sys.exit(main())
