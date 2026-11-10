import csv
import json

import numpy as np
import pandas as pd
import pytest

from data import load_daily


@pytest.fixture
def daily_rows():
    dates = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    return [
        {"open_time": int(date.timestamp() * 1000),
         "close_time": int(date.timestamp() * 1000) + 86_399_999,
         "open": 100 + i, "high": 105 + i, "low": 95 + i,
         "close": 102 + i, "volume": i,
         "open_utc": "ignored; numeric timestamps are authoritative"}
        for i, date in enumerate(dates)
    ]


def write_csv(path, rows):
    destination = path / "BTCUSDT_1d.csv"
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_manifest(path, generated_utc):
    (path / "manifest.json").write_text(
        json.dumps({"generated_utc": generated_utc}), encoding="utf-8"
    )


def test_load_daily_returns_numeric_ohlcv_with_utc_index(tmp_path, daily_rows):
    write_csv(tmp_path, daily_rows)
    frame = load_daily("BTCUSDT", csv_dir=tmp_path)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert frame.index[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert frame["close"].tolist() == [102.0, 103.0, 104.0]
    assert frame["volume"].tolist() == [0.0, 1.0, 2.0]
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes)


def test_snapshot_midday_candle_stays_incomplete_when_loaded_later(tmp_path, daily_rows):
    write_csv(tmp_path, daily_rows)
    write_manifest(tmp_path, "2020-01-03T12:00:00Z")
    current = load_daily("BTCUSDT", csv_dir=tmp_path)
    assert current["close"].tolist() == [102.0, 103.0]
    later = load_daily("BTCUSDT", csv_dir=tmp_path, as_of="2020-02-01")
    pd.testing.assert_frame_equal(current, later)


@pytest.mark.parametrize("as_of,expected", [
    ("2020-01-02T12:00:00Z", [102.0]),
    ("2020-01-02T23:59:59.999Z", [102.0]),
    ("2020-01-03T00:00:00Z", [102.0, 103.0]),
    ("2020-01-03T08:00:00+08:00", [102.0, 103.0]),
    (pd.Timestamp("2020-01-03"), [102.0, 103.0]),
])
def test_explicit_cutoff_requires_strictly_earlier_close_time(
    tmp_path, daily_rows, as_of, expected
):
    write_csv(tmp_path, daily_rows)
    assert load_daily("BTCUSDT", tmp_path, as_of=as_of)["close"].tolist() == expected


def test_explicit_cutoff_cannot_override_snapshot_date(tmp_path, daily_rows):
    write_csv(tmp_path, daily_rows)
    write_manifest(tmp_path, "2020-01-03T12:00:00Z")
    early = load_daily("BTCUSDT", tmp_path, as_of="2020-01-02T12:00:00Z")
    late = load_daily("BTCUSDT", tmp_path, as_of="2020-02-01T00:00:00Z")
    assert early["close"].tolist() == [102.0]
    assert late["close"].tolist() == [102.0, 103.0]


def test_default_now_remains_a_cutoff_even_with_future_manifest(tmp_path, daily_rows):
    future = int(pd.Timestamp("2200-01-01", tz="UTC").timestamp() * 1000)
    daily_rows[-1].update(open_time=future, close_time=future + 86_399_999)
    write_csv(tmp_path, daily_rows)
    write_manifest(tmp_path, "2201-01-01T00:00:00Z")
    assert load_daily("BTCUSDT", tmp_path)["close"].tolist() == [102.0, 103.0]


def test_adding_future_rows_does_not_change_retained_prefix(tmp_path, daily_rows):
    write_csv(tmp_path, daily_rows[:2])
    before = load_daily("BTCUSDT", tmp_path, as_of="2020-01-03T12:00:00Z")
    daily_rows[-1].update(open=1000, high=1100, low=900, close=1050)
    write_csv(tmp_path, daily_rows)
    after = load_daily("BTCUSDT", tmp_path, as_of="2020-01-03T12:00:00Z")
    pd.testing.assert_frame_equal(before, after)


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume",
                                     "open_time", "close_time"])
@pytest.mark.parametrize("value", ["broken", "NaN", "inf", "-inf"])
def test_load_daily_rejects_malformed_or_nonfinite_values(
    tmp_path, daily_rows, column, value
):
    daily_rows[1][column] = value
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError):
        load_daily("BTCUSDT", tmp_path)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("value", [0, -1])
def test_load_daily_rejects_nonpositive_prices(tmp_path, daily_rows, column, value):
    daily_rows[1][column] = value
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="price|OHLC"):
        load_daily("BTCUSDT", tmp_path)


@pytest.mark.parametrize("columns", [["open", "high", "low", "close"], ["volume"]])
def test_load_daily_rejects_boolean_fields_as_nonnumeric(tmp_path, daily_rows, columns):
    for row in daily_rows:
        for column in columns:
            row[column] = True
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="numeric|malformed"):
        load_daily("BTCUSDT", tmp_path)


def test_load_daily_rejects_negative_volume(tmp_path, daily_rows):
    daily_rows[1]["volume"] = -1
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="volume"):
        load_daily("BTCUSDT", tmp_path)


@pytest.mark.parametrize("updates", [
    {"high": 100}, {"low": 104}, {"open": 110}, {"close": 90},
    {"high": 94, "low": 106},
])
def test_load_daily_rejects_ohlc_bounds(tmp_path, daily_rows, updates):
    daily_rows[1].update(updates)
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="OHLC|bound"):
        load_daily("BTCUSDT", tmp_path)


def test_load_daily_rejects_duplicate_timestamps(tmp_path, daily_rows):
    daily_rows[1] = daily_rows[0].copy()
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="duplicate"):
        load_daily("BTCUSDT", tmp_path)


def test_load_daily_rejects_unsorted_timestamps_instead_of_sorting(tmp_path, daily_rows):
    write_csv(tmp_path, daily_rows[::-1])
    with pytest.raises(ValueError, match="monotonic|increasing|order"):
        load_daily("BTCUSDT", tmp_path)


def test_load_daily_rejects_gaps_among_retained_rows(tmp_path, daily_rows):
    write_csv(tmp_path, [daily_rows[0], daily_rows[2]])
    with pytest.raises(ValueError, match="gap"):
        load_daily("BTCUSDT", tmp_path)


@pytest.mark.parametrize("offset", [1, 0.5, 3_600_000])
def test_load_daily_rejects_non_day_aligned_open(tmp_path, daily_rows, offset):
    daily_rows[1]["open_time"] += offset
    daily_rows[1]["close_time"] += offset
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError):
        load_daily("BTCUSDT", tmp_path)


@pytest.mark.parametrize("offset", [-1, 0.5, 1, 86_400_000])
def test_load_daily_rejects_wrong_daily_close_time(tmp_path, daily_rows, offset):
    daily_rows[1]["close_time"] += offset
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="close_time|timestamp"):
        load_daily("BTCUSDT", tmp_path)


def test_load_daily_requires_close_time(tmp_path, daily_rows):
    for row in daily_rows:
        del row["close_time"]
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="close_time"):
        load_daily("BTCUSDT", tmp_path)


def test_load_daily_rejects_empty_retained_data(tmp_path, daily_rows):
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="empty|complete|retained"):
        load_daily("BTCUSDT", tmp_path, as_of="2020-01-01T12:00:00Z")


def test_load_daily_rejects_header_only_csv(tmp_path, daily_rows):
    destination = write_csv(tmp_path, daily_rows)
    destination.write_text(
        ",".join(daily_rows[0]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="empty|complete|retained"):
        load_daily("BTCUSDT", tmp_path)


@pytest.mark.parametrize("symbol", ["../BTCUSDT", "..\\BTCUSDT", "/BTCUSDT",
                                    "BTC/USDT", "BTC_USDT", "", "比特币", None])
def test_load_daily_rejects_unsafe_symbol(tmp_path, symbol):
    with pytest.raises(ValueError, match="symbol|ticker"):
        load_daily(symbol, tmp_path)


@pytest.mark.parametrize("as_of", ["not-a-date", pd.NaT, np.nan])
def test_load_daily_rejects_invalid_cutoff(tmp_path, daily_rows, as_of):
    write_csv(tmp_path, daily_rows)
    with pytest.raises(ValueError, match="as_of|cutoff"):
        load_daily("BTCUSDT", tmp_path, as_of=as_of)


@pytest.mark.parametrize("manifest", ['{"generated_utc": "bad"}',
                                      '{"generated_utc": null}', '{', '[]'])
def test_load_daily_does_not_ignore_invalid_manifest(tmp_path, daily_rows, manifest):
    write_csv(tmp_path, daily_rows)
    (tmp_path / "manifest.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|generated_utc"):
        load_daily("BTCUSDT", tmp_path)


def test_manifest_without_generated_date_uses_explicit_cutoff(tmp_path, daily_rows):
    write_csv(tmp_path, daily_rows)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    frame = load_daily("BTCUSDT", tmp_path, as_of="2020-01-02T12:00:00Z")
    assert frame["close"].tolist() == [102.0]
