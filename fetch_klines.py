#!/usr/bin/env python3
"""Fetch Binance spot klines into per-symbol/per-interval CSVs.

Source of truth is the REST mirror (data-api.binance.vision, key-free).
The monthly archives on data.binance.vision are frozen snapshots: they omit
late-arriving trades that REST has since folded into the same candles, so they
under-report volume and can miss the true high/low. They are kept behind
--source archives purely as a fast path for ranges too long to page via REST.
api.binance.com itself is unreachable on this network, hence the mirrors.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import datetime as dt
import io
import json
import os
import threading
import time
import zipfile

import requests

UTC = dt.timezone.utc
REST = "https://data-api.binance.vision/api/v3/klines"
BULK_URL = ("https://data.binance.vision/data/spot/monthly/klines/"
            "{sym}/{iv}/{sym}-{iv}-{ym}.zip")

BULK_IVS = ["5m", "15m", "1h"]
REST_IVS = ["4h", "1d", "1w", "1M"]
DERIVED_IVS = ["1y"]
FETCH_IVS = BULK_IVS + REST_IVS
PAGE = 1000
PERIOD_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000,
             "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000}
# Weekly bars open on Monday, so they are offset from the Unix epoch (a Thursday).
ALIGN_MS = {"1w": 345_600_000}

# Binance kline layout: 0 openTime 1 open 2 high 3 low 4 close 5 volume
# 6 closeTime 7 quoteVolume 8 trades 9 takerBuyBase 10 takerBuyQuote 11 ignore
HEADER = ["open_time", "open_utc", "open", "high", "low", "close", "volume",
          "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
          "close_time", "close_utc"]
CSV_TO_KLINE = [0, 2, 3, 4, 5, 6, 11, 7, 8, 9, 10]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
_local = threading.local()


def get_session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "kline-fetch/1.0"
    return _local.session


def http_get(url, params=None, timeout=90, tries=6):
    """GET with backoff. 429/418 honour Retry-After; 404 and other 4xx return as-is."""
    session, delay = get_session(), 1.0
    for _ in range(tries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if resp.status_code in (418, 429):
            time.sleep(min(int(resp.headers.get("Retry-After", delay * 4)), 120))
            delay = min(delay * 2, 60)
            continue
        if resp.status_code >= 500:
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        return resp
    raise RuntimeError(f"still failing after {tries} tries: {url}")


def to_ms(raw):
    """Archives carry microseconds, REST carries milliseconds."""
    value = int(raw)
    return value // 1000 if value > 10**14 else value


def iso(ms):
    return dt.datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M:%S")


def normalize(row):
    out = list(row)
    out[0], out[6] = to_ms(row[0]), to_ms(row[6])
    return out


def parse_zip(payload):
    archive = zipfile.ZipFile(io.BytesIO(payload))
    rows = []
    for line in archive.read(archive.namelist()[0]).decode("utf-8").splitlines():
        line = line.strip()
        if line[:1].isdigit():
            fields = line.split(",")
            if len(fields) >= 12:
                rows.append(normalize(fields))
    return rows


def fetch_month(job):
    sym, iv, year, month = job
    ym = f"{year:04d}-{month:02d}"
    resp = http_get(BULK_URL.format(sym=sym, iv=iv, ym=ym))
    if resp.status_code == 404:
        return ym, [], "404"
    resp.raise_for_status()
    return ym, parse_zip(resp.content), "ok"


def month_seq(start, stop):
    out, y, m = [], start.year, start.month
    while (y, m) < stop:
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def iter_rest(sym, iv, start_ms, end_ms, delay):
    """Yield kline rows ascending; page on close_time so calendar intervals work."""
    cur = start_ms
    while cur < end_ms:
        resp = http_get(REST, {"symbol": sym, "interval": iv, "startTime": cur,
                               "endTime": end_ms, "limit": PAGE})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            return
        for item in batch:
            yield normalize(item)
        nxt = to_ms(batch[-1][6]) + 1
        if nxt <= cur:
            return
        cur = nxt
        if len(batch) < PAGE:
            return
        time.sleep(delay)


def dedup_sort(rows):
    rows.sort(key=lambda r: r[0])
    out = []
    for r in rows:
        if not out or r[0] != out[-1][0]:
            out.append(r)
    return out


def aggregate_yearly(monthly_rows):
    """Binance exposes no 1y interval, so roll calendar months up into years."""
    groups = {}
    for r in monthly_rows:
        groups.setdefault(int(iso(r[0])[:4]), []).append(r)
    from decimal import Decimal
    yearly = []
    for year in sorted(groups):
        g = sorted(groups[year], key=lambda r: r[0])
        first, last = g[0], g[-1]
        high, low = max(g, key=lambda r: float(r[2])), min(g, key=lambda r: float(r[3]))
        sums = {i: str(sum((Decimal(r[i]) for r in g), Decimal(0))) for i in (5, 7, 9, 10)}
        yearly.append([first[0], first[1], high[2], low[3], last[4], sums[5],
                       last[6], sums[7], str(sum(int(r[8]) for r in g)),
                       sums[9], sums[10]])
    return yearly


def emit(path, rows_iter, iv, now_ms):
    """Stream rows to CSV, checking ordering/continuity/sanity as they arrive."""
    period = PERIOD_MS.get(iv)
    bars = gaps = dupes = out_of_order = ohlc_bad = 0
    prev = first = last_open = last_close = None
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        for r in rows_iter:
            o = r[0]
            if o > now_ms and period is not None:
                continue
            if prev is not None:
                if o == prev:
                    dupes += 1
                    continue
                if o < prev:
                    out_of_order += 1
                    continue
                if period and o - prev != period:
                    gaps += 1
            else:
                first = o
            op, hi, lo, cl = (float(r[i]) for i in (1, 2, 3, 4))
            if hi < max(op, cl) or lo > min(op, cl) or hi < lo:
                ohlc_bad += 1
            writer.writerow([r[0], iso(r[0]), r[1], r[2], r[3], r[4], r[5], r[7],
                             r[8], r[9], r[10], r[6], iso(r[6])])
            prev, last_open, last_close, bars = o, o, r[6], bars + 1
    if last_open is None:
        raise RuntimeError(f"no rows written for {path}")
    if period is not None:
        anchor = ALIGN_MS.get(iv, 0)
        newest = ((now_ms - anchor) // period) * period + anchor
        gaps += 1 if last_open < newest else 0
        expected = (newest - first) // period + 1
    else:
        first_y, first_m = int(iso(first)[:4]), int(iso(first)[5:7])
        now_y, now_m = int(iso(now_ms)[:4]), int(iso(now_ms)[5:7])
        expected = ((now_y * 12 + now_m) - (first_y * 12 + first_m) + 1
                    if iv == "1M" else now_y - first_y + 1)
        gaps += max(expected - bars, 0)
    return {"bars": bars, "expected": expected, "missing": expected - bars,
            "first_utc": iso(first), "last_open_utc": iso(last_open),
            "last_close_utc": iso(last_close), "gaps": gaps, "duplicates": dupes,
            "out_of_order": out_of_order, "ohlc_violations": ohlc_bad,
            "in_progress": last_close > now_ms,
            "span_days": round((last_open - first) / 86_400_000, 1)}


def worker(job):
    sym, iv, start_ms, end_ms, source, months, delay = job
    if iv in BULK_IVS and source == "archives":
        rows = []
        for y, m in months:
            _ym, chunk, status = fetch_month((sym, iv, y, m))
            if status == "ok":
                rows.extend(chunk)
        high = max((r[6] for r in rows), default=start_ms - 1)
        rows.extend(iter_rest(sym, iv, high + 1, end_ms, delay))
        return sym, iv, emit(os.path.join(OUT_DIR, f"{sym}_{iv}.csv"),
                             dedup_sort(rows), iv, end_ms)
    return sym, iv, emit(os.path.join(OUT_DIR, f"{sym}_{iv}.csv"),
                         iter_rest(sym, iv, start_ms, end_ms, delay), iv, end_ms)


def read_kline_csv(path):
    """Read a written CSV back into the Binance kline row layout emit() expects."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        rows = []
        for row in reader:
            r = [row[i] for i in CSV_TO_KLINE]
            r[0], r[6] = int(r[0]), int(r[6])
            rows.append(r)
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-11-01", help="UTC date, YYYY-MM-DD")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.12, help="pause between pages")
    ap.add_argument("--source", choices=["rest", "archives"], default="rest")
    args = ap.parse_args()

    start = dt.datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    symbols = args.symbols.split(",")
    now = dt.datetime.now(UTC).replace(second=0, microsecond=0)
    now_ms = int(now.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    months = month_seq(start, (now.year, now.month))
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = [(s, iv, start_ms, now_ms, args.source, months, args.delay)
            for s in symbols for iv in FETCH_IVS]
    manifest = {"generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source": args.source, "start_utc": start.strftime("%Y-%m-%d"),
                "columns": HEADER, "files": {}}
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sym, iv, stats in pool.map(worker, jobs):
            manifest["files"][f"{sym}_{iv}"] = stats
            flag = "OK" if stats["gaps"] == 0 else f"gaps={stats['gaps']}"
            print(f"{sym} {iv:>4}  bars={stats['bars']:>7}  "
                  f"missing={stats['missing']:>4}  {flag}  "
                  f"{stats['first_utc']} -> {stats['last_open_utc']}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    for sym in symbols:
        monthly = read_kline_csv(os.path.join(OUT_DIR, f"{sym}_1M.csv"))
        stats = emit(os.path.join(OUT_DIR, f"{sym}_1y.csv"),
                     aggregate_yearly(monthly), "1y", now_ms)
        manifest["files"][f"{sym}_1y"] = stats
        print(f"{sym}   1y  bars={stats['bars']:>7}  (rolled up from 1M)", flush=True)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    bad = [k for k, v in manifest["files"].items()
           if v["gaps"] or v["duplicates"] or v["ohlc_violations"] or v["out_of_order"]]
    print(f"\nfiles={len(manifest['files'])}  with_problems={len(bad)} {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
