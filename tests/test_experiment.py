import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def synthetic(seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-06-01", periods=900, tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, len(idx))))
    op = np.r_[100, close[:-1]]
    return pd.DataFrame({"open": op, "high": np.maximum(op, close) * 1.01,
                         "low": np.minimum(op, close) * 0.99, "close": close,
                         "volume": 1.0}, index=idx)


def dataset():
    return {"BTCUSDT": synthetic(0), "ETHUSDT": synthetic(1)}


def experiment(**kwargs):
    return importlib.import_module("run_backtest").run_experiment(**kwargs)


def test_experiment_reports_historical_validation_and_costs(tmp_path):
    result = experiment(data=dataset(), out_dir=tmp_path, plots=True)
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert {"development", "historical_validation", "full"} == set(m.segment)
    assert {"passive_risk_budget", "trend_long", "trend_dual", "short_only",
            "passive_matched_long", "passive_matched_dual"} <= set(m.strategy)
    assert {"base", "trading_costs_2x", "borrow_5pct", "borrow_20pct"} <= set(m.scenario)
    assert {"mean_gross_exposure", "fees", "slippage", "borrow", "long_pnl", "short_pnl"} <= set(m.columns)
    assert (tmp_path / "equity.png").stat().st_size > 0
    assert (tmp_path / "validation_equity.png").stat().st_size > 0
    assert (tmp_path / "exposure_trend_dual.png").stat().st_size > 0
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["fresh_holdout"] is False
    assert metadata["live_ready"] is False
    assert metadata["short_cost_model"] == "assumed_spot_borrow"
    assert metadata["parameters"]["horizons"] == [20, 60, 120]
    assert metadata["parameters"]["target_vol"] == 0.15
    assert metadata["source_hashes"]
    assert len(metadata["input_hashes"]["BTCUSDT"]) == 64
    assert result.output_dir == tmp_path
    for strategy in ("passive_risk_budget", "trend_long", "trend_dual"):
        equity = pd.read_csv(tmp_path / f"equity_{strategy}_validation.csv", index_col=0).iloc[:, 0]
        ledger = pd.read_csv(tmp_path / f"ledger_{strategy}_validation.csv", index_col=0)
        assert equity.iloc[0] == 20000
        assert equity.diff().fillna(0).to_numpy() == pytest.approx((ledger.long_pnl + ledger.short_pnl).to_numpy())
    long_rows = m[(m.strategy == "trend_long") & (m.scenario == "base")]
    assert long_rows.borrow.tolist() == [0.0] * len(long_rows)
    assert long_rows.short_pnl.to_numpy() == pytest.approx(np.zeros(len(long_rows)))


def test_calibration_and_development_results_ignore_future_prices(tmp_path):
    data = dataset()
    before = experiment(data=data, out_dir=tmp_path / "before", plots=False)
    changed = {s: df.copy() for s, df in data.items()}
    for df in changed.values():
        df.loc["2023":, ["open", "high", "low", "close"]] *= 1.8
    after = experiment(data=changed, out_dir=tmp_path / "after", plots=False)
    assert before.metadata["risk_match_factors"] == after.metadata["risk_match_factors"]
    pd.testing.assert_frame_equal(before.metrics[before.metrics.segment == "development"],
                                  after.metrics[after.metrics.segment == "development"])
    assert not (tmp_path / "after" / "equity.png").exists()
    assert not (tmp_path / "after" / "scan_A.csv").exists()


def test_output_directory_never_overwrites_existing_artifacts(tmp_path):
    marker = tmp_path / "existing.csv"
    marker.write_text("keep this", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment(data=dataset(), out_dir=tmp_path, plots=False)
    assert marker.read_text(encoding="utf-8") == "keep this"


def test_different_symbol_dates_fail_instead_of_forward_filling(tmp_path):
    data = dataset()
    data["ETHUSDT"] = data["ETHUSDT"].iloc[1:]
    with pytest.raises(ValueError, match="align"):
        experiment(data=data, out_dir=tmp_path, plots=False)


def test_borrow_table_requires_complete_daily_coverage(tmp_path):
    module = importlib.import_module("run_backtest")
    path = tmp_path / "borrow.csv"
    path.write_text("date,symbol,annual_rate\n2023-01-01,BTCUSDT,0.10\n", encoding="utf-8")
    idx = pd.date_range("2023-01-01", periods=2, tz="UTC")
    with pytest.raises(ValueError, match="coverage"):
        module.load_borrow_rates(path, idx, ["BTCUSDT", "ETHUSDT"])


def test_borrow_table_preserves_observed_rates(tmp_path):
    module = importlib.import_module("run_backtest")
    path = tmp_path / "borrow.csv"
    path.write_text("date,symbol,annual_rate\n2023-01-01,BTCUSDT,0.10\n2023-01-01,ETHUSDT,0.20\n", encoding="utf-8")
    idx = pd.date_range("2023-01-01", periods=1, tz="UTC")
    rates = module.load_borrow_rates(path, idx, ["BTCUSDT", "ETHUSDT"])
    assert rates.iloc[0].tolist() == [0.10, 0.20]


def test_cli_exposes_reproducible_research_inputs():
    script = Path(__file__).resolve().parents[1] / "run_backtest.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for option in ("--csv-dir", "--out-dir", "--development-end", "--borrow-rates", "--no-plots", "--as-of"):
        assert option in result.stdout


def test_output_occupied_during_simulation_is_not_overwritten(tmp_path, monkeypatch):
    module = importlib.import_module("run_backtest")
    marker = tmp_path / "other_run.csv"
    real = module.run_portfolio
    state = {"calls": 0}

    def occupying(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            marker.write_text("other run", encoding="utf-8")
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "run_portfolio", occupying)
    with pytest.raises(FileExistsError):
        module.run_experiment(data=dataset(), out_dir=tmp_path, plots=False)
    assert marker.read_text(encoding="utf-8") == "other run"
    assert not (tmp_path / "metrics.csv").exists()


def test_concurrent_second_run_is_refused_before_writing(tmp_path, monkeypatch):
    module = importlib.import_module("run_backtest")
    real = module.run_portfolio
    state = {"calls": 0}

    def guarded(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            with pytest.raises(FileExistsError):
                module.run_experiment(data=dataset(), out_dir=tmp_path, plots=False)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "run_portfolio", guarded)
    module.run_experiment(data=dataset(), out_dir=tmp_path, plots=False)
    assert (tmp_path / "metrics.csv").exists()
