import pandas as pd
import pytest

import backtest


def test_flattening_and_reentry_are_not_blocked_by_trading_buffer():
    idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    prices = pd.DataFrame({"BTCUSDT": [100.0] * 5}, index=idx)
    target = pd.DataFrame({"BTCUSDT": [0.01, 0, 0.01, 0, 0]}, index=idx)
    result = backtest.run_portfolio(prices, prices, target,
                                    costs=backtest.Costs(0, 0, 0),
                                    initial_cash=10000, rebalance_band=0.02)
    assert result.fills.delta_qty.tolist() == pytest.approx([1, -1, 1, -1])
    assert result.equity.tolist() == pytest.approx([10000] * 5)
