import math

import numpy as np
import pandas as pd
import pytest

import signals


def prices(log_returns, start="2020-01-01"):
    returns = np.asarray(log_returns, dtype=float)
    if returns.ndim == 1:
        returns = returns[:, None]
    levels = np.vstack([np.zeros(returns.shape[1]), returns.cumsum(axis=0)])
    return pd.DataFrame(
        100 * np.exp(levels),
        index=pd.date_range(start, periods=len(levels), freq="D", tz="UTC"),
        columns=[f"asset_{i}" for i in range(returns.shape[1])],
    )


def forecasts(close, value=1.0):
    return pd.DataFrame(value, index=close.index, columns=close.columns)


def test_trend_forecast_hand_calculated_sample_vol_and_clipped_components():
    close = prices([0.1, -0.1, 0.2])
    result = signals.trend_forecast(close, horizons=(1, 2, 3), vol_window=3)
    # Returns are .1, -.1, .2: sample variance = 7/300.
    vol = math.sqrt(7 / 300)
    expected = (1 + 0.1 / (vol * math.sqrt(2))
                + 0.2 / (vol * math.sqrt(3))) / 3
    assert result.iloc[:3].isna().all().all()
    assert result.iloc[3, 0] == pytest.approx(expected)
    assert result.index.equals(close.index)
    assert result.columns.equals(close.columns)


def test_trend_forecast_waits_for_every_horizon_and_full_return_window():
    close = prices([0.02, -0.01, 0.03, -0.02, 0.04, -0.03])
    long_horizon = signals.trend_forecast(close, horizons=(1, 5), vol_window=3)
    long_vol = signals.trend_forecast(close, horizons=(1, 3), vol_window=5)
    assert long_horizon.iloc[:5].isna().all().all()
    assert long_vol.iloc[:5].isna().all().all()
    assert np.isfinite(long_horizon.iloc[5:]).all().all()
    assert np.isfinite(long_vol.iloc[5:]).all().all()


def test_trend_default_windows_first_available_after_120_days():
    close = prices(np.tile([0.01, -0.02, 0.03], 50))
    result = signals.trend_forecast(close)
    assert result.iloc[:120].isna().all().all()
    assert np.isfinite(result.iloc[120:]).all().all()
    assert (result.iloc[120:].abs() <= 1).all().all()


def test_trend_clips_both_signs_and_uses_current_close_without_execution_shift():
    close = prices([[0.01, -0.01], [-0.01, 0.01], [0.10, -0.10]])
    result = signals.trend_forecast(close, horizons=(1,), vol_window=3)
    assert result.iloc[-1].tolist() == pytest.approx([1.0, -1.0])


@pytest.mark.parametrize("growth", [0.0, 0.01, -0.01, math.log(2)])
def test_trend_zero_volatility_does_not_amplify_deterministic_growth(growth):
    close = prices([growth] * 12)
    result = signals.trend_forecast(close, horizons=(1, 3), vol_window=4)
    assert result.iloc[:4].isna().all().all()
    np.testing.assert_array_equal(result.iloc[4:].to_numpy(), 0.0)


@pytest.mark.parametrize("growth", [0.0, 0.01, -0.01, math.log(2)])
def test_trend_zero_volatility_after_nonconstant_history(growth):
    close = prices([0.1, -0.1, 0.2, -0.2] + [growth] * 12)
    result = signals.trend_forecast(close, horizons=(1, 3), vol_window=4)
    np.testing.assert_array_equal(result.iloc[8:].to_numpy(), 0.0)


def test_trend_is_prefix_invariant_including_non_month_end_final_row():
    close = prices(np.tile([0.03, -0.01, -0.02, 0.01], 40), start="2020-01-20")
    full = signals.trend_forecast(close)
    for length in (121, 132, 143, len(close)):
        prefix = signals.trend_forecast(close.iloc[:length])
        pd.testing.assert_frame_equal(prefix, full.iloc[:length])
    changed = close.copy()
    changed.iloc[132:] *= np.linspace(1.5, 5, len(close) - 132)[:, None]
    pd.testing.assert_frame_equal(
        signals.trend_forecast(changed).iloc[:132], full.iloc[:132]
    )


def test_portfolio_hand_calculated_equal_risk_covariance_weights():
    # Orthogonal centered returns: annual sigma A=sqrt(365/75), B=2*A.
    close = prices([[-0.1, -0.2], [0.1, -0.2], [-0.1, 0.2], [0.1, 0.2]])
    forecast = forecasts(close)
    result = signals.portfolio_weights(close, forecast, vol_window=4)
    sigma = math.sqrt(365 / 75)  # sample variance .04 / 3, annualized 365
    expected_a = 0.15 / (sigma * math.sqrt(2))
    assert (result.iloc[:4] == 0).all().all()
    assert result.iloc[4].tolist() == pytest.approx([expected_a, expected_a / 2])
    assert result.index.equals(close.index)
    assert result.columns.equals(close.columns)


def test_portfolio_default_covariance_requires_60_complete_returns():
    close = prices(np.tile([0.01, -0.02, 0.03], 25))
    result = signals.portfolio_weights(close, forecasts(close))
    assert (result.iloc[:60] == 0).all().all()
    assert (result.iloc[60:] > 0).all().all()


def test_portfolio_covariance_accounts_for_correlation():
    independent = prices([[-0.1, -0.1], [0.1, -0.1], [-0.1, 0.1], [0.1, 0.1]])
    correlated = prices([[-0.1, -0.1], [0.1, 0.1], [-0.1, -0.1], [0.1, 0.1]])
    ind = signals.portfolio_weights(independent, forecasts(independent), vol_window=4)
    corr = signals.portfolio_weights(correlated, forecasts(correlated), vol_window=4)
    assert ind.iloc[-1].sum() == pytest.approx(corr.iloc[-1].sum() * math.sqrt(2))
    assert corr.iloc[-1, 0] == pytest.approx(0.15 / (2 * math.sqrt(365 / 75)))


def test_portfolio_annual_vol_floor_limits_inverse_vol_imbalance():
    close = prices([[-0.0001, -0.01], [0.0001, -0.01],
                    [-0.0001, 0.01], [0.0001, 0.01]])
    result = signals.portfolio_weights(
        close, forecasts(close), vol_window=4, max_gross=10, max_asset=10
    )
    sigma_b = math.sqrt(365 / 7500)
    assert result.iloc[-1, 0] / result.iloc[-1, 1] == pytest.approx(sigma_b / 0.05)


def test_decreasing_forecast_strength_decreases_exposure_without_rescaling_up():
    close = prices([[-0.1, -0.2], [0.1, -0.2], [-0.1, 0.2], [0.1, 0.2]])
    strong = signals.portfolio_weights(close, forecasts(close), vol_window=4)
    weak = signals.portfolio_weights(close, forecasts(close, 0.25), vol_window=4)
    pd.testing.assert_frame_equal(weak, strong * 0.25)
    assert weak.iloc[-1].abs().sum() < strong.iloc[-1].abs().sum()


def test_passive_purelong_and_dual_use_the_same_forecast_weight_interface():
    close = prices([[-0.1, -0.2], [0.1, -0.2], [-0.1, 0.2], [0.1, 0.2]])
    signed = forecasts(close)
    signed.iloc[:, 0] = -0.5
    signed.iloc[:, 1] = 0.25
    passive = signals.portfolio_weights(close, forecasts(close), vol_window=4)
    purelong = signals.portfolio_weights(close, signed.clip(lower=0), vol_window=4)
    dual = signals.portfolio_weights(close, signed, vol_window=4)
    assert purelong.iloc[-1].tolist() == pytest.approx([0, passive.iloc[-1, 1] * 0.25])
    assert dual.iloc[-1].tolist() == pytest.approx(
        [-passive.iloc[-1, 0] * 0.5, passive.iloc[-1, 1] * 0.25]
    )


def test_portfolio_asset_and_gross_caps_hold():
    close = prices([[0.001, 0.001], [-0.001, -0.001]] * 5)
    result = signals.portfolio_weights(
        close, forecasts(close), vol_window=4, max_asset=0.2, max_gross=0.3
    )
    assert (result.abs() <= 0.2 + 1e-12).all().all()
    assert (result.abs().sum(axis=1) <= 0.3 + 1e-12).all()
    assert result.iloc[-1].tolist() == pytest.approx([0.15, 0.15])


def test_portfolio_scales_down_signed_forecasts_that_remove_a_covariance_hedge():
    close = prices([[-0.02, 0.01], [-0.01, 0.02], [0.01, -0.02], [0.02, -0.01]])
    forecast = forecasts(close)
    forecast.iloc[:, 1] = -1.0
    result = signals.portfolio_weights(
        close, forecast, vol_window=4, max_asset=10, max_gross=10
    )
    # Sample annual covariance: diag=365/3000, off-diag=-.8*diag.
    cov = (365 / 3000) * np.array([[1.0, -0.8], [-0.8, 1.0]])
    weights = result.iloc[-1].to_numpy()
    assert weights[0] > 0 > weights[1]
    assert math.sqrt(weights @ cov @ weights) == pytest.approx(0.15)


@pytest.mark.parametrize("growth", [0.0, 0.01, math.log(2)])
def test_portfolio_zero_covariance_never_gets_positive_exposure(growth):
    close = prices([[growth, -growth]] * 10)
    result = signals.portfolio_weights(close, forecasts(close), vol_window=4)
    np.testing.assert_array_equal(result.to_numpy(), 0.0)


def test_portfolio_zero_risk_hedge_returns_zero_weights():
    close = prices([[-0.1, -0.1], [0.1, 0.1], [-0.1, -0.1], [0.1, 0.1]])
    forecast = forecasts(close)
    forecast.iloc[:, 1] = -1.0
    result = signals.portfolio_weights(close, forecast, vol_window=4)
    np.testing.assert_array_equal(result.to_numpy(), 0.0)


def test_portfolio_warmup_or_missing_forecast_produces_zero_entire_row():
    close = prices([[0.03, -0.01], [-0.02, 0.04], [0.01, 0.02], [-0.01, -0.03]] * 3)
    forecast = forecasts(close)
    forecast.iloc[4, 0] = np.nan
    result = signals.portfolio_weights(close, forecast, vol_window=4)
    assert (result.iloc[:5] == 0).all().all()
    assert (result.iloc[5] > 0).all()


@pytest.mark.parametrize("parameter", ["max_asset", "max_gross"])
def test_portfolio_zero_cap_means_no_exposure(parameter):
    close = prices([[0.03, -0.01], [-0.02, 0.04], [0.01, 0.02], [-0.01, -0.03]] * 3)
    result = signals.portfolio_weights(
        close, forecasts(close), vol_window=4, **{parameter: 0}
    )
    np.testing.assert_array_equal(result.to_numpy(), 0.0)


def test_weights_are_prefix_invariant_and_future_prices_cannot_change_the_past():
    returns = np.column_stack([
        np.tile([0.01, -0.03, 0.02, 0.01], 45),
        np.tile([-0.03, 0.02, 0.04, -0.01], 45),
    ])
    close = prices(returns, start="2020-01-20")
    forecast = signals.trend_forecast(close)
    full = signals.portfolio_weights(close, forecast)
    assert full.iloc[130].abs().sum() > 0
    for length in (61, 121, 132, 147):
        part = close.iloc[:length]
        pd.testing.assert_frame_equal(
            signals.portfolio_weights(part, signals.trend_forecast(part)),
            full.iloc[:length],
        )
    changed = close.copy()
    changed.iloc[132:] *= np.linspace(1, 10, len(close) - 132)[:, None]
    changed_forecast = signals.trend_forecast(changed)
    pd.testing.assert_frame_equal(
        signals.portfolio_weights(changed, changed_forecast).iloc[:132], full.iloc[:132]
    )


@pytest.mark.parametrize("api", ["trend", "weights"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, "100", True, 1j])
def test_signal_apis_reject_nonpositive_nonfinite_or_nonnumeric_closes(api, value):
    close = prices([0.02, -0.01, 0.03]).astype(object)
    close.iloc[2, 0] = value
    with pytest.raises(ValueError, match="close|price|numeric|finite"):
        if api == "trend":
            signals.trend_forecast(close)
        else:
            signals.portfolio_weights(close, forecasts(close))


@pytest.mark.parametrize("api", ["trend", "weights"])
@pytest.mark.parametrize("defect", ["duplicate", "reverse", "gap", "intraday",
                                    "nat", "not_dates", "columns"])
def test_signal_apis_reject_invalid_daily_axes(api, defect):
    close = prices([[0.01, -0.03], [-0.02, 0.02], [0.03, -0.01]])
    if defect == "duplicate":
        close.index = close.index[:2].append(close.index[1:3])
    elif defect == "reverse":
        close = close.iloc[::-1]
    elif defect == "gap":
        close = close.drop(close.index[1])
    elif defect == "intraday":
        close.index += pd.Timedelta(hours=1)
    elif defect == "nat":
        close.index = pd.DatetimeIndex([pd.NaT, *close.index[1:]])
    elif defect == "not_dates":
        close.index = pd.RangeIndex(len(close))
    else:
        close.columns = ["duplicate", "duplicate"]
    with pytest.raises(ValueError):
        if api == "trend":
            signals.trend_forecast(close)
        else:
            signals.portfolio_weights(close, forecasts(close))


@pytest.mark.parametrize("horizons", [(), (0,), (-1,), (1.5,), (True,), "20", 20, None])
def test_trend_rejects_invalid_horizons(horizons):
    with pytest.raises(ValueError, match="horizon"):
        signals.trend_forecast(prices([0.01, -0.02]), horizons=horizons)


@pytest.mark.parametrize("api", ["trend", "weights"])
@pytest.mark.parametrize("window", [0, -1, 1.5, True, None, np.nan])
def test_signal_apis_require_positive_integer_vol_window(api, window):
    close = prices([0.01, -0.02])
    with pytest.raises(ValueError, match="vol_window"):
        if api == "trend":
            signals.trend_forecast(close, vol_window=window)
        else:
            signals.portfolio_weights(close, forecasts(close), vol_window=window)


def test_single_return_window_has_no_sample_volatility():
    close = prices([0.01, -0.02, 0.03])
    assert signals.trend_forecast(close, horizons=(1,), vol_window=1).isna().all().all()
    result = signals.portfolio_weights(close, forecasts(close), vol_window=1)
    np.testing.assert_array_equal(result.to_numpy(), 0.0)


@pytest.mark.parametrize("defect", ["index", "columns", "column_order", "not_frame"])
def test_portfolio_requires_exact_forecast_alignment(defect):
    close = prices([[0.01, -0.03], [-0.02, 0.02]])
    forecast = forecasts(close)
    if defect == "index":
        forecast.index += pd.Timedelta(days=1)
    elif defect == "columns":
        forecast.columns = ["other", "asset_1"]
    elif defect == "column_order":
        forecast = forecast.iloc[:, ::-1]
    else:
        forecast = forecast.iloc[:, 0]
    with pytest.raises(ValueError, match="forecast|match"):
        signals.portfolio_weights(close, forecast)


@pytest.mark.parametrize("value", [1.01, -1.01, np.inf, -np.inf, "0.5", True, 1j])
def test_portfolio_rejects_invalid_forecasts(value):
    close = prices([0.01, -0.02, 0.03])
    forecast = forecasts(close).astype(object)
    forecast.iloc[-1, 0] = value
    with pytest.raises(ValueError, match="forecast"):
        signals.portfolio_weights(close, forecast)


@pytest.mark.parametrize("parameter,value", [
    ("target_vol", 0), ("target_vol", -0.1), ("target_vol", 1.01),
    ("target_vol", np.nan), ("target_vol", np.inf), ("target_vol", True),
    ("max_gross", -1), ("max_gross", np.inf), ("max_gross", np.nan),
    ("max_asset", -1), ("max_asset", np.inf), ("max_asset", "0.6"),
])
def test_portfolio_rejects_invalid_risk_parameters(parameter, value):
    close = prices([0.01, -0.02])
    with pytest.raises(ValueError, match=parameter):
        signals.portfolio_weights(close, forecasts(close), **{parameter: value})


def test_signal_apis_do_not_mutate_inputs():
    close = prices([[0.03, -0.01], [-0.02, 0.04], [0.01, 0.02], [-0.01, -0.03]] * 3)
    before_close = close.copy(deep=True)
    forecast = signals.trend_forecast(close, horizons=(1, 2), vol_window=4)
    before_forecast = forecast.copy(deep=True)
    signals.portfolio_weights(close, forecast, vol_window=4)
    pd.testing.assert_frame_equal(close, before_close)
    pd.testing.assert_frame_equal(forecast, before_forecast)
