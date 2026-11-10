"""Frozen-parameter portfolio research on previously inspected market history."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import Costs, PortfolioResult, run_portfolio
from data import OHLCV, load_daily
from metrics import _utc_timestamp, annual_returns, perf_stats, rolling_stats
from report import equity_chart, exposure_chart
from signals import portfolio_weights, trend_forecast

SYMBOLS = ("BTCUSDT", "ETHUSDT")
PARAMETERS = {"horizons": [20, 60, 120], "vol_window": 60, "target_vol": 0.15,
              "max_gross": 1.0, "max_asset": 0.6, "rebalance_band": 0.02,
              "initial_cash": 20000.0}
PRIMARY = ("passive_risk_budget", "trend_long", "trend_dual", "short_only")


@dataclass
class ExperimentResult:
    output_dir: Path
    metrics: pd.DataFrame
    metadata: dict


def load_borrow_rates(path, index, symbols) -> pd.DataFrame:
    table = pd.read_csv(path)
    if not {"date", "symbol", "annual_rate"}.issubset(table.columns):
        raise ValueError("borrow table requires date, symbol, annual_rate columns")
    table["date"] = pd.to_datetime(table["date"], utc=True, errors="raise")
    if (table["date"].isna().any() or table["symbol"].isna().any()
            or table.duplicated(["date", "symbol"]).any()
            or not table["date"].equals(table["date"].dt.normalize())):
        raise ValueError("borrow table requires unique daily date/symbol observations")
    if pd.api.types.is_bool_dtype(table["annual_rate"].dtype):
        raise ValueError("borrow annual rates must be numeric")
    table["annual_rate"] = pd.to_numeric(table["annual_rate"], errors="raise")
    if not np.isfinite(table["annual_rate"]).all() or (table["annual_rate"] < 0).any():
        raise ValueError("borrow annual rates must be finite and nonnegative")
    rates = table.pivot(index="date", columns="symbol", values="annual_rate")
    rates = rates.reindex(index=index, columns=symbols)
    if rates.isna().any().any():
        raise ValueError("borrow table needs complete daily date/symbol coverage")
    return rates.astype(float)


def _validate_data(data):
    if set(data) != set(SYMBOLS):
        raise ValueError("research requires BTCUSDT and ETHUSDT")
    index = data[SYMBOLS[0]].index
    for symbol in SYMBOLS:
        frame = data[symbol]
        if not frame.index.equals(index):
            raise ValueError("symbol dates must align exactly; missing bars are not filled")
        if (not isinstance(index, pd.DatetimeIndex) or str(index.tz) != "UTC"
                or index.hasnans or not index.is_unique or not index.is_monotonic_increasing
                or not index.equals(index.normalize())
                or not ((index[1:] - index[:-1]) == pd.Timedelta(days=1)).all()):
            raise ValueError("data requires consecutive midnight UTC dates")
        if frame.empty or not set(OHLCV).issubset(frame.columns):
            raise ValueError(f"{symbol}: nonempty OHLCV data is required")
        if any(not pd.api.types.is_numeric_dtype(dtype)
               or pd.api.types.is_bool_dtype(dtype)
               or pd.api.types.is_complex_dtype(dtype) for dtype in frame[OHLCV].dtypes):
            raise ValueError(f"{symbol}: OHLCV must be real numeric values")
        if (not np.isfinite(frame[OHLCV].to_numpy(dtype=float)).all()
                or (frame[["open", "high", "low", "close"]] <= 0).any().any()
                or (frame.volume < 0).any()
                or (frame.high < frame[["open", "low", "close"]].max(axis=1)).any()
                or (frame.low > frame[["open", "high", "close"]].min(axis=1)).any()):
            raise ValueError(f"{symbol}: invalid OHLCV observations")


def _summary(strategy, scenario, segment, result, end=None):
    stats = perf_stats(result.equity, end=end)
    ledger = result.ledger.iloc[1:]
    if end is not None:
        ledger = ledger.loc[:end]
    return {"strategy": strategy, "scenario": scenario, "segment": segment, **stats,
            **{key: float(ledger[key].sum()) for key in
               ("fees", "slippage", "borrow", "long_pnl", "short_pnl")},
            "mean_gross_exposure": float(ledger.gross_exposure.mean()),
            "peak_gross_exposure": float(ledger.gross_exposure.max()),
            "turnover_annual": float(ledger.turnover.sum() * 365 / len(ledger)),
            "n_fills": int(ledger.n_fills.sum())}


def _save_result(out, strategy, scope, scenario, result, targets):
    suffix = f"{strategy}_{scope}" + (f"_{scenario}" if scenario != "base" else "")
    for name in ("equity", "weights", "ledger", "asset_pnl", "fills"):
        getattr(result, name).to_csv(out / f"{name}_{suffix}.csv", index=name != "fills")
    targets.loc[result.equity.index].to_csv(out / f"targets_{suffix}.csv")


def _frame_hash(frame):
    digest = hashlib.sha256()
    digest.update(str(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def run_experiment(data=None, csv_dir="csv", out_dir=None,
                   development_end="2022-12-31", plots=True, borrow_rates=None,
                   as_of=None) -> ExperimentResult:
    stamp = pd.Timestamp.now(tz="UTC")
    out = Path(out_dir) if out_dir is not None else Path("results_v2") / stamp.strftime("%Y%m%dT%H%M%S%fZ")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"refusing to overwrite nonempty output: {out}")
    loaded = data is None
    if loaded:
        data = {symbol: load_daily(symbol, csv_dir, as_of=as_of) for symbol in SYMBOLS}
    else:
        data = {symbol: frame.copy(deep=True) for symbol, frame in data.items()}
        if as_of is not None:
            cutoff = _utc_timestamp(as_of)
            data = {symbol: frame.loc[frame.index + pd.Timedelta(days=1) <= cutoff]
                    for symbol, frame in data.items()}
    _validate_data(data)
    opens = pd.DataFrame({symbol: data[symbol].open for symbol in SYMBOLS})
    closes = pd.DataFrame({symbol: data[symbol].close for symbol in SYMBOLS})
    boundary = _utc_timestamp(development_end)
    if boundary != boundary.normalize():
        raise ValueError("development_end must be midnight UTC")
    forecast = trend_forecast(closes, horizons=PARAMETERS["horizons"],
                              vol_window=PARAMETERS["vol_window"])
    ready = forecast.notna().all(axis=1)
    if not ready.any():
        raise ValueError("insufficient history for the fixed forecast warmup")
    start = ready[ready].index[0]
    if boundary not in closes.index or boundary <= start or boundary >= closes.index[-1]:
        raise ValueError("development_end must leave returns in both development and validation")
    sizing = {key: PARAMETERS[key] for key in ("vol_window", "target_vol", "max_gross", "max_asset")}
    forecasts = {"passive_risk_budget": pd.DataFrame(1.0, index=closes.index, columns=closes.columns),
                 "trend_long": forecast.clip(lower=0), "trend_dual": forecast,
                 "short_only": forecast.clip(upper=0)}
    targets = {name: portfolio_weights(closes, signal, **sizing) for name, signal in forecasts.items()}
    opens, closes = opens.loc[start:], closes.loc[start:]
    observed_rates = None
    borrow_hash = None
    if borrow_rates is not None:
        if isinstance(borrow_rates, pd.DataFrame):
            observed_rates = borrow_rates.copy(deep=True)
            if (not observed_rates.index.equals(opens.index)
                    or not observed_rates.columns.equals(opens.columns)):
                raise ValueError("borrow rates must align with the post-warmup execution dates")
            borrow_hash = _frame_hash(observed_rates)
        else:
            observed_rates = load_borrow_rates(borrow_rates, opens.index, list(SYMBOLS))
            borrow_hash = hashlib.sha256(Path(borrow_rates).read_bytes()).hexdigest()
    execution = {key: PARAMETERS[key] for key in ("initial_cash", "rebalance_band", "max_gross", "max_asset")}

    # Exclusive reservation so a concurrent run fails instead of mixing artifacts.
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(stamp.isoformat())
    except FileExistsError as exc:
        raise FileExistsError(f"output directory already claimed by another run: {out}") from exc

    def simulate(name, scope, costs=Costs(), assumed_borrow=False):
        dates = opens.index if scope == "full" else opens.loc[boundary:].index
        rates = None if observed_rates is None or assumed_borrow else observed_rates.loc[dates]
        return run_portfolio(opens.loc[dates], closes.loc[dates], targets[name].loc[dates],
                             costs=costs, borrow_rates=rates, **execution)

    runs: dict[tuple[str, str, str], PortfolioResult] = {}
    rows = []
    for name in PRIMARY:
        result = simulate(name, "full")
        runs[(name, "full", "base")] = result
        rows.append(_summary(name, "base", "development", result, end=boundary))
        rows.append(_summary(name, "base", "full", result))
    passive_vol = perf_stats(runs[("passive_risk_budget", "full", "base")].equity,
                             end=boundary)["vol_annual"]
    risk_match_factors = {}
    for direction in ("long", "dual"):
        trend_vol = perf_stats(runs[(f"trend_{direction}", "full", "base")].equity,
                               end=boundary)["vol_annual"]
        if passive_vol is None or passive_vol <= 0 or trend_vol is None:
            raise ValueError("development history cannot calibrate passive risk matching")
        factor = min(1.0, trend_vol / passive_vol)
        risk_match_factors[direction] = factor
        targets[f"passive_matched_{direction}"] = targets["passive_risk_budget"] * factor

    for name in targets:
        result = simulate(name, "validation")
        runs[(name, "validation", "base")] = result
        rows.append(_summary(name, "base", "historical_validation", result))
    for name in PRIMARY:
        result = simulate(name, "validation", Costs(fee=0.002, slippage=0.001))
        runs[(name, "validation", "trading_costs_2x")] = result
        rows.append(_summary(name, "trading_costs_2x", "historical_validation", result))
    for name in ("trend_dual", "short_only"):
        for scenario, rate in (("borrow_5pct", 0.05), ("borrow_20pct", 0.20)):
            result = simulate(name, "validation", Costs(borrow_annual=rate), assumed_borrow=True)
            runs[(name, "validation", scenario)] = result
            rows.append(_summary(name, scenario, "historical_validation", result))

    source_dir = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                     for name in ("data.py", "signals.py", "backtest.py", "metrics.py", "report.py", "run_backtest.py")}
    manifest_path = Path(csv_dir) / "manifest.json"
    snapshot = json.loads(manifest_path.read_text(encoding="utf-8")) if loaded and manifest_path.exists() else None
    metadata = {
        "generated_utc": stamp.isoformat(), "parameters": PARAMETERS.copy(),
        "base_costs": asdict(Costs()), "development_end": boundary.isoformat(),
        "input_start": data[SYMBOLS[0]].index[0].isoformat(),
        "last_complete_bar": closes.index[-1].isoformat(), "initial_nav_date": start.isoformat(),
        "as_of": _utc_timestamp(as_of).isoformat() if as_of is not None else None,
        "snapshot": snapshot, "fresh_holdout": False, "live_ready": False,
        "short_cost_model": "supplied_daily_spot_borrow" if observed_rates is not None else "assumed_spot_borrow",
        "borrow_input_hash": borrow_hash, "risk_match_factors": risk_match_factors,
        "risk_match_method": "development realized volatility ratio; capped at 1; frozen for validation only",
        "validation_method": "fresh equal-capital accounts at development_end; next-open execution with past warmup",
        "signal_timing": "close t signal executes at open t+1; NAV dated by the daily bar open date",
        "borrow_timing": "EOD short market value times annual rate / 365",
        "scenarios": {"base": "default trading costs and supplied or assumed borrow",
                      "trading_costs_2x": "double fee and slippage; base borrowing unchanged",
                      "borrow_5pct": "hypothetical flat 5% annual borrowing",
                      "borrow_20pct": "hypothetical flat 20% annual borrowing"},
        "source_hashes": source_hashes,
        "input_hashes": {symbol: _frame_hash(data[symbol][OHLCV]) for symbol in SYMBOLS},
        "limitations": ["Previously inspected history, not a fresh holdout or proof of alpha.",
                        "Only two correlated surviving assets; no universe selection correction.",
                        "Synthetic spot shorts; borrowing availability and forced recall are not modeled.",
                        "No perpetual funding, margin liquidation, intraday stops, or market impact model.",
                        "Idle cash earns zero; terminal positions are marked, not liquidated.",
                        "Execution caps may be exceeded by subsequent price and borrowing-cost drift.",
                        "Development-calibrated passive exposure need not match future realized risk.",
                        "Short-only is an ablation, not the short contribution of the dual portfolio."]}
    metrics = pd.DataFrame(rows)
    if any(entry.name != lock.name for entry in out.iterdir()):
        raise FileExistsError(f"output directory occupied during run: {out}")
    metrics.to_csv(out / "metrics.csv", index=False)
    forecast.to_csv(out / "forecast.csv")
    annual, rolling = [], []
    for (name, scope, scenario), result in runs.items():
        _save_result(out, name, scope, scenario, result, targets[name])
        if scenario == "base":
            annual.append(annual_returns(result.equity).assign(strategy=name, segment=scope))
            rolling.append(rolling_stats(result.equity).assign(strategy=name, segment=scope)
                           .rename_axis("date").reset_index())
    pd.concat(annual, ignore_index=True).to_csv(out / "annual_returns.csv", index=False)
    pd.concat(rolling, ignore_index=True).to_csv(out / "rolling_365.csv", index=False)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    if plots:
        equity_chart({name: runs[(name, "full", "base")].equity for name in PRIMARY},
                     out / "equity.png", "Frozen trend research: full history", vline=boundary)
        equity_chart({name: runs[(name, "validation", "base")].equity for name in targets},
                     out / "validation_equity.png", "Historical validation (previously inspected)")
        for name in PRIMARY:
            exposure_chart(runs[(name, "full", "base")].weights, out / f"exposure_{name}.png",
                           f"{name}: realized end-of-day exposure")
    lock.unlink()
    return ExperimentResult(out, metrics, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default="csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--development-end", default="2022-12-31")
    parser.add_argument("--borrow-rates", help="CSV with date,symbol,annual_rate and complete daily coverage")
    parser.add_argument("--as-of", help="UTC completed-bar cutoff, also limited by acquisition manifest")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    result = run_experiment(csv_dir=args.csv_dir, out_dir=args.out_dir,
                            development_end=args.development_end, borrow_rates=args.borrow_rates,
                            as_of=args.as_of, plots=not args.no_plots)
    selected = result.metrics[(result.metrics.segment == "historical_validation")
                              & (result.metrics.scenario == "base")]
    print(selected[["strategy", "cagr", "vol_annual", "max_dd", "sharpe", "mean_gross_exposure"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Research-only artifacts: {result.output_dir.resolve()}")


if __name__ == "__main__":
    main()