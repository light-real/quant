import numpy as np
import pandas as pd
import pytest

import run_regime
import signals


def prices(log_returns, start="2020-01-01", columns=None):
    returns = np.asarray(log_returns, dtype=float)
    if returns.ndim == 1:
        returns = returns[:, None]
    levels = np.vstack([np.zeros(returns.shape[1]), returns.cumsum(axis=0)])
    names = columns or [f"asset_{i}" for i in range(returns.shape[1])]
    return pd.DataFrame(
        100 * np.exp(levels),
        index=pd.date_range(start, periods=len(levels), freq="D", tz="UTC"),
        columns=names,
    )


def test_regime_forecast_stays_flat_through_the_warmup():
    close = prices(np.full(30, 0.01))
    result = signals.regime_forecast(close, ma_len=10)
    assert (result.iloc[:9] == 0).all().all()
    assert (result.iloc[10:] == 1).all().all()
    assert result.index.equals(close.index) and result.columns.equals(close.columns)


def test_regime_forecast_turns_off_when_price_loses_the_average():
    up = np.full(20, 0.02)
    down = np.full(20, -0.03)
    result = signals.regime_forecast(prices(np.concatenate([up, down])), ma_len=10)
    assert result.iloc[19, 0] == 1.0
    assert result.iloc[-1, 0] == 0.0


def test_regime_forecast_hysteresis_holds_state_inside_the_band():
    # 40 flat sessions, 30 at 120 (regime on), then a pullback to 115 while the
    # 20-day average is 119.75: below it, but inside the 5% band (lower 113.76).
    path = np.concatenate([np.full(40, 100.0), np.full(30, 120.0), [115.0]])
    close = pd.DataFrame(
        {"asset_0": path},
        index=pd.date_range("2020-01-01", periods=len(path), freq="D", tz="UTC"),
    )
    assert close.rolling(20).mean().iloc[-1, 0] == pytest.approx(119.75)
    plain = signals.regime_forecast(close, ma_len=20, hysteresis=0.0)
    banded = signals.regime_forecast(close, ma_len=20, hysteresis=0.05)
    assert plain.iloc[-1, 0] == 0.0
    assert banded.iloc[-1, 0] == 1.0
    assert banded.iloc[40:-1, 0].eq(1.0).all()


def test_regime_forecast_is_causal_per_row():
    rng = np.random.default_rng(7)
    close = prices(rng.normal(0, 0.02, size=(120, 1)))
    baseline = signals.regime_forecast(close, ma_len=20)
    shocked = close.copy()
    shocked.iloc[80:] = shocked.iloc[80:] * 1.5
    changed = signals.regime_forecast(shocked, ma_len=20)
    assert baseline.iloc[:80].equals(changed.iloc[:80])


def test_regime_weights_sizes_only_the_active_asset():
    rng = np.random.default_rng(3)
    close = prices(rng.normal(0, 0.02, size=(200, 2)), columns=["a", "b"])
    regime = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    regime["a"] = 1.0
    only_a = signals.regime_weights(close, regime, vol_window=30, target_vol=0.30,
                                    max_gross=2.0, max_asset=1.0)
    assert (only_a["b"] == 0).all()
    assert only_a["a"].iloc[40:].gt(0).all()

    regime["b"] = 1.0
    both = signals.regime_weights(close, regime, vol_window=30, target_vol=0.30,
                                  max_gross=2.0, max_asset=1.0)
    # The single active asset must not be half-sized as if a dormant asset existed.
    assert only_a["a"].iloc[40] > both["a"].iloc[40]


def test_regime_weights_respects_caps_and_empty_regimes():
    rng = np.random.default_rng(11)
    close = prices(rng.normal(0.01, 0.05, size=(200, 2)), columns=["a", "b"])
    flat = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    empty = signals.regime_weights(close, flat, vol_window=30, target_vol=0.9,
                                   max_gross=1.0, max_asset=0.6)
    assert (empty == 0).all().all()

    full = signals.regime_weights(close, flat + 1.0, vol_window=30, target_vol=0.9,
                                  max_gross=1.0, max_asset=0.6)
    assert (full.abs() <= 0.6 + 1e-12).all().all()
    assert (full.abs().sum(axis=1) <= 1.0 + 1e-12).all()


def test_regime_weights_rejects_misaligned_or_non_binary_regime():
    close = prices(np.full(50, 0.01))
    with pytest.raises(ValueError):
        signals.regime_weights(close, pd.DataFrame(1.0, index=close.index[:-1],
                                                   columns=close.columns))
    with pytest.raises(ValueError):
        signals.regime_weights(close, pd.DataFrame(2.0, index=close.index,
                                                   columns=close.columns))


def test_regime_forecast_validates_arguments():
    close = prices(np.full(20, 0.01))
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError):
            signals.regime_forecast(close, ma_len=bad)
    for bad in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError):
            signals.regime_forecast(close, ma_len=5, hysteresis=bad)


def test_targets_for_leaves_untraded_symbols_flat():
    rng = np.random.default_rng(5)
    close = prices(rng.normal(0.005, 0.03, size=(300, 2)), columns=["BTCUSDT", "ETHUSDT"])
    weights = run_regime.targets_for(close, ma_len=60, hysteresis=0.0)
    assert (weights["ETHUSDT"] == 0).all()
    assert weights["BTCUSDT"].abs().gt(0).any()
    with pytest.raises(ValueError):
        run_regime.targets_for(close, ma_len=60, hysteresis=0.0, traded=("DOGEUSDT",))


def test_select_parameters_takes_the_highest_development_sharpe():
    scores = pd.DataFrame({"ma_len": [60, 80, 100], "hysteresis": [0.0, 0.0, 0.02],
                           "development_sharpe": [0.9, 1.4, 1.1]})
    assert run_regime.select_parameters(scores) == {"ma_len": 80, "hysteresis": 0.0}
    with pytest.raises(ValueError):
        run_regime.select_parameters(scores.iloc[0:0])


def test_walk_forward_trains_only_before_the_test_year(monkeypatch):
    rng = np.random.default_rng(13)
    close = prices(rng.normal(0.004, 0.02, size=(1500, 1)), columns=["BTCUSDT"],
                   start="2020-01-01")
    opens = close * 1.001
    seen = []

    original = run_regime.development_scores

    def spy(opens_arg, closes_arg, boundary):
        seen.append((opens_arg.index.min(), opens_arg.index.max(), boundary))
        return original(opens_arg, closes_arg, boundary)

    monkeypatch.setattr(run_regime, "development_scores", spy)
    table = run_regime.walk_forward(opens, close, boundary=pd.Timestamp("2022-12-31", tz="UTC"),
                                    years=(2023,))
    assert len(table) == 1 and table.iloc[0]["year"] == 2023
    assert table.iloc[0]["ma_len"] in run_regime.GRID["ma_len"]
    start, _, boundary = seen[0]
    assert start >= pd.Timestamp("2020-01-01", tz="UTC")
    assert boundary == pd.Timestamp("2022-12-31", tz="UTC")
