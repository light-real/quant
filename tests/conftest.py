import warnings

warnings.filterwarnings("ignore", message=".*numexpr.*")
warnings.filterwarnings("ignore", message=".*bottleneck.*")

import pandas as pd
import pytest

CSV_HEADER = ["open_time", "open_utc", "open", "high", "low", "close", "volume",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
              "close_time", "close_utc"]


@pytest.fixture
def ohlc():
    def _make(closes, opens=None, highs=None, lows=None, start="2020-01-01"):
        n = len(closes)
        idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
        o = list(opens) if opens is not None else list(closes)
        h = list(highs) if highs is not None else [max(a, b) for a, b in zip(o, closes)]
        l = list(lows) if lows is not None else [min(a, b) for a, b in zip(o, closes)]
        return pd.DataFrame({"open": o, "high": h, "low": l,
                             "close": list(closes), "volume": [1.0] * n}, index=idx)
    return _make


@pytest.fixture
def write_daily_csv():
    def _write(path, symbol, rows):
        idx = pd.date_range("2020-01-01", periods=len(rows), freq="D", tz="UTC")
        lines = [",".join(CSV_HEADER)]
        for i, (o, h, l, c) in enumerate(rows):
            ms = int(idx[i].timestamp() * 1000)
            iso = idx[i].strftime("%Y-%m-%dT%H:%M:%S")
            lines.append(f"{ms},{iso},{o},{h},{l},{c},1,1,1,1,1,{ms + 86399999},{iso}")
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{symbol}_1d.csv").write_text("\n".join(lines), encoding="utf-8")
    return _write