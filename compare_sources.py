#!/usr/bin/env python3
"""Bidirectional diff of the monthly archives against the REST mirror.

Shows which source holds candles the other lacks, and where the two disagree on
the same candle. Run this before trusting archive data for a range.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import datetime as dt
import sys

from fetch_klines import fetch_month, month_seq

UTC = dt.timezone.utc
# archive row (Binance layout) -> CSV column name in the written files
FIELD_MAP = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}


def when(ms):
    return dt.datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%dT%H:%M")


def describe(archive_row, csv_row):
    if archive_row is not None:
        return (f"o={archive_row[1]:>13} h={archive_row[2]:>13} "
                f"l={archive_row[3]:>13} vol={archive_row[5]:>15}")
    return (f"o={csv_row['open']:>13} h={csv_row['high']:>13} "
            f"l={csv_row['low']:>13} vol={csv_row['volume']:>15}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    end = dt.datetime.now(UTC)
    months = month_seq(dt.datetime.fromisoformat(args.start).replace(tzinfo=UTC),
                       (end.year, end.month))
    archive = {}
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(args.symbol, args.interval, y, m) for (y, m) in months]
        for ym, rows, status in pool.map(fetch_month, jobs):
            if status != "ok":
                print(f"  archive {ym}: HTTP {status}")
                continue
            for row in rows:
                archive[row[0]] = row
    rest = {}
    with open(f"csv/{args.symbol}_{args.interval}.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rest[int(row["open_time"])] = row

    print(f"archive: {len(archive):>8} bars over {len(months)} monthly files "
          f"(ends {when(max(archive))})")
    print(f"rest:    {len(rest):>8} bars (ends {when(max(rest))})")

    only_archive = sorted(set(archive) - set(rest))
    only_rest = sorted(set(rest) - set(archive))
    print(f"\ncandles present in the archive but absent from REST: {len(only_archive)}")
    for t in only_archive[:args.show]:
        print(f"   {when(t)}  {describe(archive[t], None)}")
    print(f"candles present in REST but absent from the archive:   {len(only_rest)}")
    for t in only_rest[:args.show]:
        print(f"   {when(t)}  {describe(None, rest[t])}")

    disagree = {name: 0 for name in FIELD_MAP}
    worst = []
    for t in set(archive) & set(rest):
        for name, idx in FIELD_MAP.items():
            av, rv = float(archive[t][idx]), float(rest[t][name])
            if abs(av - rv) > 1e-6 * (abs(rv) + 1):
                disagree[name] += 1
                if name == "volume":
                    worst.append((abs(av - rv) / max(abs(rv), 1.0), t, av, rv))
    shared = len(set(archive) & set(rest))
    print(f"\nover {shared} shared candles, the two sources disagree on: {disagree}")
    worst.sort(reverse=True)
    print("largest volume gaps (archive vs REST):")
    for rel, t, av, rv in worst[:args.show]:
        print(f"   {when(t)}  archive={av:>16,.5f}  rest={rv:>16,.5f}  gap={rel:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
