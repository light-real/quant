import math

import numpy as np
import pandas as pd
import pytest

import metrics


def equity(values, start="2020-12-31"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"), dtype=float)


def test_perf_stats_counts_first_selected_loss_and_uses_sample_volatility():
    # Returns are -10%, +10%: compounded -1%, drawdown 10%, mean zero.
    result = metrics.perf_stats(equity([100, 90, 99]), start="2021-01-01")
    assert result == pytest.approx({
        "n_days": 2,
        "total_return": -0.01,
        "cagr": 0.99 ** (365 / 2) - 1,
        "vol_annual": math.sqrt(7.3),
        "sharpe": 0,
        "max_dd": 0.1,
        "calmar": (0.99 ** (365 / 2) - 1) / 0.1,
    }, abs=1e-12)


def test_perf_stats_default_includes_all_returns_not_initial_snapshot():
    # +25%, -20%, +25%: total +25%, peak-to-trough loss 20%.
    result = metrics.perf_stats(equity([100, 125, 100, 125]))
    assert result == pytest.approx({
        "n_days": 3,
        "total_return": 0.25,
        "cagr": 1.25 ** (365 / 3) - 1,
        "vol_annual": math.sqrt(24.6375),
        "sharpe": math.sqrt(365) * 0.1 / math.sqrt(0.0675),
        "max_dd": 0.2,
        "calmar": (1.25 ** (365 / 3) - 1) / 0.2,
    })


def test_perf_stats_bounds_are_inclusive_and_rebase_selected_drawdown():
    # Prior 50% loss is outside selection; selected returns are +10%, -10%.
    result = metrics.perf_stats(
        equity([200, 100, 110, 99, 198]), "2021-01-02", "2021-01-03"
    )
    assert result["n_days"] == 2
    assert result["total_return"] == pytest.approx(-0.01)
    assert result["max_dd"] == pytest.approx(0.1)


def test_perf_stats_single_selected_loss_has_undefined_sample_volatility():
    result = metrics.perf_stats(equity([100, 90, 99]), end="2021-01-01")
    assert result["n_days"] == 1
    assert result["total_return"] == pytest.approx(-0.1)
    assert result["cagr"] == pytest.approx(0.9 ** 365 - 1)
    assert result["max_dd"] == pytest.approx(0.1)
    assert result["calmar"] == pytest.approx((0.9 ** 365 - 1) / 0.1)
    assert result["vol_annual"] is None
    assert result["sharpe"] is None


@pytest.mark.parametrize("start,end", [
    (None, "2020-12-31"),
    ("2022-01-01", None),
    ("2019-01-01", "2019-02-01"),
])
def test_perf_stats_empty_selection_has_explicit_no_observations(start, end):
    result = metrics.perf_stats(equity([100, 90, 99]), start, end)
    assert result == {"n_days": 0, "total_return": None, "cagr": None,
                      "vol_annual": None, "sharpe": None, "max_dd": None, "calmar": None}


def test_perf_stats_initial_snapshot_alone_is_not_a_return():
    result = metrics.perf_stats(equity([100]))
    assert result == {"n_days": 0, "total_return": None, "cagr": None,
                      "vol_annual": None, "sharpe": None, "max_dd": None, "calmar": None}


def test_perf_stats_cash_ratios_are_undefined_not_zero():
    result = metrics.perf_stats(equity([100, 100, 100]))
    assert result == {"n_days": 2, "total_return": 0, "cagr": 0,
                      "vol_annual": 0, "sharpe": None, "max_dd": 0, "calmar": None}


def test_perf_stats_no_drawdown_has_undefined_calmar():
    result = metrics.perf_stats(equity([100, 125, 250]))
    assert result["max_dd"] == 0
    assert result["calmar"] is None
    assert result["sharpe"] == pytest.approx(math.sqrt(365) * 0.625 / math.sqrt(0.28125))


def test_perf_stats_accepts_naive_and_aware_date_bounds_without_mutation():
    nav = equity([100, 90, 99])
    original = nav.copy(deep=True)
    result = metrics.perf_stats(nav, pd.Timestamp("2021-01-01"), "2021-01-02T01:00:00+01:00")
    assert result["n_days"] == 2
    assert result["total_return"] == pytest.approx(-0.01)
    pd.testing.assert_series_equal(nav, original)


@pytest.mark.parametrize("start,end", [
    ("not-a-date", None), (None, "not-a-date"),
    (pd.NaT, None), (None, pd.NaT),
    ("2021-01-02", "2021-01-01"), (123, None),
])
def test_perf_stats_rejects_bad_bounds(start, end):
    with pytest.raises(ValueError):
        metrics.perf_stats(equity([100, 90, 99]), start, end)


def test_annual_returns_keeps_january_first_and_only_observed_return_years():
    result = metrics.annual_returns(equity([100, 90, 99]))
    pd.testing.assert_frame_equal(result, pd.DataFrame({
        "year": [2021], "return": [-0.01], "n_days": [2],
    }))


def test_annual_returns_partial_first_year_and_cross_year_compounding():
    result = metrics.annual_returns(equity([100, 110, 99, 108.9], start="2020-12-30"))
    pd.testing.assert_frame_equal(result, pd.DataFrame({
        "year": [2020, 2021], "return": [0.1, -0.01], "n_days": [1, 2],
    }))


def test_annual_returns_leap_year_uses_actual_daily_count():
    result = metrics.annual_returns(equity([100] * 368, start="2019-12-31"))
    pd.testing.assert_frame_equal(result, pd.DataFrame({
        "year": [2020, 2021], "return": [0.0, 0.0], "n_days": [366, 1],
    }))


def test_annual_returns_snapshot_only_is_empty_with_schema():
    result = metrics.annual_returns(equity([100]))
    assert result.empty
    assert list(result.columns) == ["year", "return", "n_days"]


def test_rolling_stats_requires_full_return_window_and_sample_std():
    nav = equity([100, 90, 99, 99])
    result = metrics.rolling_stats(nav, window=2)
    pd.testing.assert_index_equal(result.index, nav.index)
    assert list(result.columns) == ["rolling_return", "rolling_vol", "rolling_sharpe"]
    assert result.iloc[:2].isna().all().all()
    assert result.iloc[2].to_dict() == pytest.approx({
        "rolling_return": -0.01, "rolling_vol": math.sqrt(7.3), "rolling_sharpe": 0,
    }, abs=1e-12)
    assert result.iloc[3].to_dict() == pytest.approx({
        "rolling_return": 0.1, "rolling_vol": math.sqrt(1.825),
        "rolling_sharpe": math.sqrt(182.5),
    })


def test_rolling_stats_has_no_future_dependency():
    nav = equity([100, 90, 99, 99, 1, 1000])
    before = metrics.rolling_stats(nav.iloc[:4], window=2)
    after = metrics.rolling_stats(nav, window=2).iloc[:4]
    pd.testing.assert_frame_equal(before, after)


def test_rolling_stats_default_365_days_and_cash_sharpe_nan():
    result = metrics.rolling_stats(equity([100] * 367))
    assert result.iloc[:365].isna().all().all()
    assert result["rolling_return"].iloc[365:].tolist() == [0, 0]
    assert result["rolling_vol"].iloc[365:].tolist() == [0, 0]
    assert result["rolling_sharpe"].isna().all()


def test_rolling_stats_short_history_is_all_nan():
    result = metrics.rolling_stats(equity([100]))
    assert result.shape == (1, 3)
    assert result.isna().all().all()


@pytest.mark.parametrize("window", [0, 1, -1, 2.5, True, "2", None])
def test_rolling_stats_rejects_invalid_window(window):
    with pytest.raises(ValueError):
        metrics.rolling_stats(equity([100, 90, 99]), window)


BAD_EQUITIES = [
    pytest.param(equity([]), id="no-initial-snapshot"),
    pytest.param(equity([100, 0]), id="zero"),
    pytest.param(equity([100, -1]), id="negative"),
    pytest.param(equity([100, np.nan]), id="nan"),
    pytest.param(equity([100, np.inf]), id="infinite"),
    pytest.param(equity([100, 90]).iloc[::-1], id="descending"),
    pytest.param(pd.Series([100, 90], index=pd.to_datetime(["2021-01-01"] * 2, utc=True)), id="duplicate"),
    pytest.param(pd.Series([100, 90], index=pd.to_datetime(["2021-01-01", "2021-01-03"], utc=True)), id="missing-day"),
    pytest.param(pd.Series([100, 90], index=pd.date_range("2021-01-01", periods=2)), id="naive"),
    pytest.param(pd.Series([100, 90], index=pd.date_range("2021-01-01", periods=2, tz="US/Eastern")), id="non-utc"),
    pytest.param(pd.Series([100, 90], index=pd.date_range("2021-01-01T12:00", periods=2, tz="UTC")), id="not-midnight"),
    pytest.param(pd.Series([100, 90], index=pd.date_range("2021-01-01", periods=2, freq="h", tz="UTC")), id="intraday"),
    pytest.param(pd.Series([100, 90], index=pd.to_datetime(["2021-01-01", None], utc=True)), id="nat"),
    pytest.param(pd.Series([100, 90]), id="not-datetime-index"),
    pytest.param(pd.Series(["100", "bad"], index=equity([100, 90]).index), id="nonnumeric"),
    pytest.param(pd.Series([100 + 1j, 90 + 1j], index=equity([100, 90]).index), id="complex"),
]


@pytest.mark.parametrize("function", ["perf_stats", "annual_returns", "rolling_stats"])
@pytest.mark.parametrize("nav", BAD_EQUITIES)
def test_metrics_reject_invalid_equity(function, nav):
    with pytest.raises(ValueError):
        getattr(metrics, function)(nav)


def test_perf_stats_validates_full_equity_even_when_bad_price_outside_selection():
    with pytest.raises(ValueError):
        metrics.perf_stats(equity([100, 90, 99, np.nan]), end="2021-01-02")
