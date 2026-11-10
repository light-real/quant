"""Regime-switch research: trade only while price holds above a long average.

Motivation, measured on the same daily history the other experiment uses:

* BTC/ETH daily returns are fat tailed (BTC excess kurtosis ~22) and their
  volatility switches regime by a factor of four (30-day realized vol runs
  from ~27% at the 5th percentile to ~100% at the 95th).
* The continuous momentum forecast in signals.trend_forecast has no robust
  daily-horizon edge: variance ratios sit near 1.0 and the profitable bucket
  is only the strongly positive one.
* A binary long/flat switch on a long moving average does hold up, but only on
  BTC. ETH shows no stable average-length plateau during development, so it is
  deliberately not traded here.

Selection is development-only. The validation window is evaluated once with
frozen parameters, and a rolling walk-forward re-selects the parameters every
calendar year to show that the choice is not a single lucky cell.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import Costs, PortfolioResult, run_portfolio
from data import load_daily
from metrics import annual_returns, perf_stats
from report import equity_chart
from signals import regime_forecast, regime_weights

SYMBOLS = ("BTCUSDT", "ETHUSDT")
TRADED = ("BTCUSDT",)
DEVELOPMENT_END = "2022-12-31"
SIZING = {"vol_window": 60, "target_vol": 0.60, "max_gross": 1.8, "max_asset": 0.9}
EXECUTION = {"initial_cash": 20000.0, "rebalance_band": 0.15, "max_gross": 1.8,
             "max_asset": 0.9}
GRID = {"ma_len": (60, 80, 100, 120, 140, 160), "hysteresis": (0.0, 0.02, 0.05)}
WALK_FORWARD_YEARS = (2023, 2024, 2025, 2026)


def _costs(**overrides) -> Costs:
    base = {"fee": Costs().fee, "slippage": Costs().slippage,
            "borrow_annual": Costs().borrow_annual,
            "cash_annual": 0.025, "margin_annual": 0.08}
    base.update(overrides)
    return Costs(**base)


def _boundary(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp(DEVELOPMENT_END if value is None else value)
    return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")


def load_frames(csv_dir="csv", as_of=None):
    data = {symbol: load_daily(symbol, csv_dir, as_of=as_of) for symbol in SYMBOLS}
    opens = pd.DataFrame({symbol: data[symbol].open for symbol in SYMBOLS})
    closes = pd.DataFrame({symbol: data[symbol].close for symbol in SYMBOLS})
    return opens, closes


def targets_for(closes: pd.DataFrame, ma_len: int, hysteresis: float,
                traded=TRADED) -> pd.DataFrame:
    """Risk-targeted weights for the traded subset; everything else stays flat."""
    unknown = set(traded) - set(closes.columns)
    if unknown:
        raise ValueError(f"unknown traded symbols: {sorted(unknown)}")
    regime = regime_forecast(closes, ma_len=ma_len, hysteresis=hysteresis)
    weights = regime_weights(closes, regime, **SIZING)
    for column in weights.columns:
        if column not in traded:
            weights[column] = 0.0
    return weights


def passive_targets(closes: pd.DataFrame, traded=TRADED) -> pd.DataFrame:
    """Always-in risk-targeted benchmark on the same traded subset."""
    regime = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for column in traded:
        regime[column] = 1.0
    return regime_weights(closes, regime, **SIZING)


def _slice(opens, closes, targets, start, end):
    if start is None and end is None:
        return opens, closes, targets
    mask = pd.Series(True, index=opens.index)
    if start is not None:
        mask &= opens.index >= start
    if end is not None:
        mask &= opens.index <= end
    return opens.loc[mask], closes.loc[mask], targets.loc[mask]


def simulate(opens, closes, targets, start=None, end=None,
             costs: Costs | None = None) -> PortfolioResult:
    """Run one fresh account over an inclusive date window."""
    o, c, w = _slice(opens, closes, targets, start, end)
    if len(o) < 2:
        raise ValueError("window too short to simulate")
    return run_portfolio(o, c, w, costs=_costs() if costs is None else costs, **EXECUTION)


def development_scores(opens, closes, boundary) -> pd.DataFrame:
    """Development Sharpe for every grid cell. Validation data is not touched."""
    rows = []
    for ma_len in GRID["ma_len"]:
        for hysteresis in GRID["hysteresis"]:
            result = simulate(opens, closes, targets_for(closes, ma_len, hysteresis),
                              end=boundary)
            stats = perf_stats(result.equity, end=boundary)
            rows.append({"ma_len": ma_len, "hysteresis": hysteresis,
                         "development_sharpe": stats["sharpe"],
                         "development_cagr": stats["cagr"]})
    return pd.DataFrame(rows)


def select_parameters(scores: pd.DataFrame) -> dict:
    """Highest development Sharpe wins; ties break toward the shorter average."""
    ranked = scores.sort_values(["development_sharpe", "ma_len"],
                                ascending=[False, True])
    if ranked.empty or not np.isfinite(ranked.iloc[0]["development_sharpe"]):
        raise ValueError("development scores are unusable")
    top = ranked.iloc[0]
    return {"ma_len": int(top["ma_len"]), "hysteresis": float(top["hysteresis"])}


def sensitivity(opens, closes, boundary, chosen: dict) -> pd.DataFrame:
    """Development and validation Sharpe side by side, for every grid cell."""
    rows = []
    for ma_len in GRID["ma_len"]:
        for hysteresis in GRID["hysteresis"]:
            targets = targets_for(closes, ma_len, hysteresis)
            dev = perf_stats(simulate(opens, closes, targets, end=boundary).equity, end=boundary)
            val = perf_stats(simulate(opens, closes, targets,
                                      start=boundary + pd.Timedelta(days=1)).equity)
            rows.append({"ma_len": ma_len, "hysteresis": hysteresis,
                         "development_sharpe": dev["sharpe"],
                         "validation_sharpe": val["sharpe"],
                         "validation_cagr": val["cagr"],
                         "validation_max_dd": val["max_dd"]})
    table = pd.DataFrame(rows)
    table["selected"] = (table.ma_len == chosen["ma_len"]) & (
        table.hysteresis == chosen["hysteresis"])
    return table


def walk_forward(opens, closes, boundary, years=WALK_FORWARD_YEARS) -> pd.DataFrame:
    """Re-select parameters at each year start and score the following year.

    Unlike the one-shot validation, walk-forward deliberately trains on every
    observation before the test year, including the original validation window.
    Each year is therefore out of sample only with respect to its own selection;
    it demonstrates parameter stability, not an untouched holdout.
    """
    del boundary  # the rolling design uses all history strictly before each year
    rows = []
    for year in years:
        train_end = pd.Timestamp(f"{year - 1}-12-31", tz="UTC")
        test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        test_end = pd.Timestamp(f"{year}-12-31", tz="UTC")
        scores = development_scores(opens.loc[:test_start], closes.loc[:test_start],
                                    train_end)
        chosen = select_parameters(scores)
        targets = targets_for(closes, chosen["ma_len"], chosen["hysteresis"])
        result = simulate(opens, closes, targets, start=test_start, end=test_end)
        stats = perf_stats(result.equity)
        rows.append({"year": year, "ma_len": chosen["ma_len"],
                     "hysteresis": chosen["hysteresis"],
                     "train_sharpe": float(scores.development_sharpe.max()),
                     "oos_sharpe": stats["sharpe"], "oos_cagr": stats["cagr"],
                     "oos_vol": stats["vol_annual"], "oos_max_dd": stats["max_dd"]})
    return pd.DataFrame(rows)


def _summary(name, segment, result: PortfolioResult) -> dict:
    stats = perf_stats(result.equity)
    ledger = result.ledger.iloc[1:]
    return {"strategy": name, "segment": segment, **stats,
            "mean_gross": float(ledger.gross_exposure.mean()),
            "turnover_annual": float(ledger.turnover.sum() * 365 / len(ledger)),
            "fees": float(ledger.fees.sum()), "financing": float(ledger.financing.sum()),
            "n_fills": int(ledger.n_fills.sum())}


def run_regime(csv_dir="csv", out_dir=None, development_end=None, plots=True,
               as_of=None) -> dict:
    boundary = _boundary(development_end)
    opens, closes = load_frames(csv_dir, as_of=as_of)
    if boundary not in opens.index:
        raise ValueError("development_end must be an observed session date")
    scores = development_scores(opens, closes, boundary)
    chosen = select_parameters(scores)
    targets = targets_for(closes, chosen["ma_len"], chosen["hysteresis"])
    benchmark = passive_targets(closes)

    development = simulate(opens, closes, targets, end=boundary)
    full = simulate(opens, closes, targets)
    validation = simulate(opens, closes, targets,
                          start=boundary + pd.Timedelta(days=1))
    validation_passive = simulate(opens, closes, benchmark,
                                  start=boundary + pd.Timedelta(days=1))

    rows = [_summary("regime", "development", development),
            _summary("regime", "full", full),
            _summary("regime", "validation", validation),
            _summary("btc_passive_same_sizing", "validation", validation_passive)]
    metrics = pd.DataFrame(rows)
    years = annual_returns(validation.equity)
    table = sensitivity(opens, closes, boundary, chosen)
    forward = walk_forward(opens, closes, boundary)

    stamp = pd.Timestamp.now(tz="UTC")
    out = (Path(out_dir) if out_dir is not None
           else Path("results_v3") / f"regime_{stamp.strftime('%Y%m%dT%H%M%SZ')}")
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    years.to_csv(out / "annual_returns.csv")
    table.to_csv(out / "sensitivity.csv", index=False)
    forward.to_csv(out / "walk_forward.csv", index=False)
    validation.equity.to_csv(out / "equity_validation.csv")
    validation.weights.to_csv(out / "weights_validation.csv")
    validation.ledger.to_csv(out / "ledger_validation.csv")
    if plots:
        equity_chart({"regime": validation.equity,
                      "btc_passive": validation_passive.equity},
                     out / "equity_validation.png",
                     "Validation gallery: regime vs passive BTC")
    source_dir = Path(__file__).resolve().parent
    metadata = {
        "generated_utc": stamp.isoformat(),
        "parameters": {**chosen, **SIZING, **EXECUTION,
                       "traded": list(TRADED), "development_end": boundary.isoformat()},
        "base_costs": asdict(_costs()),
        "grid": {key: list(value) for key, value in GRID.items()},
        "selection_rule": "highest development Sharpe over the grid; validation evaluated once",
        "walk_forward_years": list(WALK_FORWARD_YEARS),
        "signal_timing": "close t regime executes at open t+1",
        "excluded_assets": {"ETHUSDT": "no stable moving-average plateau in development"},
        "source_hashes": {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                          for name in ("signals.py", "backtest.py", "metrics.py",
                                       "data.py", "run_regime.py")},
        "limitations": [
            "single traded asset: the strategy is long BTC or in cash, nothing else",
            "no funding, basis or liquidation modelling beyond a flat margin rate",
            "development window contains one bull and one bear market only",
            "walk-forward re-selects from the same small grid each year",
        ],
        "fresh_holdout": False,
        "live_ready": False,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"output_dir": out, "chosen": chosen, "metrics": metrics,
            "sensitivity": table, "walk_forward": forward, "annual_returns": years}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", default="csv")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--development-end", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    outcome = run_regime(csv_dir=args.csv_dir, out_dir=args.out_dir,
                         development_end=args.development_end,
                         plots=not args.no_plots)
    chosen = outcome["chosen"]
    print(f"chosen: MA{chosen['ma_len']} hysteresis {chosen['hysteresis']:.2f}")
    print(outcome["metrics"].to_string(index=False))
    print(outcome["walk_forward"].to_string(index=False))
    print(f"artifacts: {outcome['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
