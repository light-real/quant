import numpy as np
import pandas as pd
import pytest

import backtest


FREE = backtest.Costs(fee=0.0, slippage=0.0, borrow_annual=0.0)


def market(opens, closes=None, weights=None):
    idx = pd.date_range("2020-01-30", periods=len(opens), tz="UTC")
    op = pd.DataFrame(opens, index=idx, dtype=float)
    cl = pd.DataFrame(closes if closes is not None else opens, index=idx, dtype=float)
    target = pd.DataFrame(weights, index=idx, columns=op.columns, dtype=float)
    return op, cl, target


def run(op, cl, target, **kwargs):
    return backtest.run_portfolio(op, cl, target, initial_cash=10000,
                                  max_asset=1.0, costs=kwargs.pop("costs", FREE), **kwargs)


def test_signal_executes_next_open_and_accounts_for_overnight_exit():
    op, cl, target = market([[100], [110], [120]], [[100], [121], [120]],
                            [[0.5], [0], [0]])
    result = run(op, cl, target)
    assert result.fills.iloc[0].time == op.index[1]
    assert result.fills.iloc[0].price == 110
    assert result.equity.tolist() == pytest.approx([10000, 10500, 10454.5454545])


def test_same_direction_target_change_rebalances_quantity():
    op, cl, target = market([[100]] * 4, weights=[[0.2], [0.8], [0], [0]])
    result = run(op, cl, target)
    assert result.fills.delta_qty.tolist() == pytest.approx([20, 60, -80])
    assert result.weights.iloc[:, 0].tolist() == pytest.approx([0, 0.2, 0.8, 0])
    assert result.equity.tolist() == pytest.approx([10000] * 4)


def test_post_cost_sizing_and_two_sided_slippage():
    op, cl, target = market([[100]] * 3, weights=[[1], [0], [0]])
    result = run(op, cl, target, costs=backtest.Costs(0.001, 0.01, 0.0))
    assert result.fills.price.tolist() == pytest.approx([101, 99])
    assert result.equity.iloc[-1] == pytest.approx(10000 * 98.901 / 101.101)
    assert result.weights.iloc[1, 0] == pytest.approx(1)
    assert result.ledger.fees.sum() > 0
    assert result.ledger.slippage.sum() > 0


def test_unchanged_weights_do_not_generate_full_turnover_or_monthly_fees():
    op, cl, target = market([[100, 100]] * 5, weights=[[0.5, 0.5]] * 5)
    result = run(op, cl, target, costs=backtest.Costs(0.001, 0, 0))
    assert len(result.fills) == 2
    assert result.equity.iloc[1:].tolist() == pytest.approx([10000 / 1.001] * 4)


def test_two_assets_share_one_capital_pool():
    op, cl, target = market([[100, 100], [100, 100]],
                            [[100, 100], [110, 90]], [[0.5, 0.5], [0, 0]])
    result = run(op, cl, target)
    assert result.fills.delta_qty.tolist() == pytest.approx([50, 50])
    assert result.equity.tolist() == pytest.approx([10000, 10000])
    assert result.asset_pnl.iloc[1].tolist() == pytest.approx([500, -500])


def test_borrow_is_in_equity_and_short_attribution():
    op, cl, target = market([[100]] * 3, weights=[[-0.5], [0], [0]])
    result = run(op, cl, target, costs=backtest.Costs(0, 0, 0.10))
    cost = 5000 * 0.10 / 365
    assert result.equity.iloc[-1] == pytest.approx(10000 - cost)
    assert result.ledger.borrow.sum() == pytest.approx(cost)
    assert result.ledger.short_pnl.sum() == pytest.approx(-cost)
    assert result.asset_pnl.iloc[:, 0].sum() == pytest.approx(-cost)


def test_supplied_daily_borrow_rates_override_flat_assumption():
    op, cl, target = market([[100]] * 3, weights=[[-0.5], [0], [0]])
    rates = pd.DataFrame([[0.1], [0.365], [0.9]], index=op.index, columns=op.columns)
    result = run(op, cl, target, borrow_rates=rates)
    assert result.ledger.borrow.sum() == pytest.approx(5.0)
    assert result.equity.iloc[-1] == pytest.approx(9995.0)


def test_long_to_short_reversal_attributes_gap_and_intraday_separately():
    op, cl, target = market([[100], [100], [90]], [[100], [110], [80]],
                            [[0.5], [-0.5], [0]])
    result = run(op, cl, target)
    assert result.ledger.long_pnl.iloc[2] == pytest.approx(-1000)
    assert result.ledger.short_pnl.iloc[2] == pytest.approx(9500 / 90 * 0.5 * 10)
    assert result.equity.iloc[-1] == pytest.approx(10027.7777778)


def test_all_costs_reconcile_daily_and_by_asset():
    op, cl, target = market([[100, 50], [101, 49], [98, 54], [97, 52]],
                            [[100, 50], [102, 48], [99, 55], [95, 53]],
                            [[0.3, -0.2], [-0.4, 0.3], [0, 0], [0, 0]])
    result = run(op, cl, target, costs=backtest.Costs())
    pnl = result.equity.diff().fillna(0)
    assert result.asset_pnl.sum(axis=1).to_numpy() == pytest.approx(pnl.to_numpy())
    assert (result.ledger.long_pnl + result.ledger.short_pnl).to_numpy() == pytest.approx(pnl.to_numpy())
    assert result.ledger.fees.sum() == pytest.approx(result.fills.fee.sum())
    assert result.ledger.slippage.sum() == pytest.approx(result.fills.slippage.sum())


def test_trading_buffer_avoids_small_same_sign_adjustments():
    op, cl, target = market([[100]] * 4, weights=[[0.5], [0.51], [0.53], [0.53]])
    result = run(op, cl, target, rebalance_band=0.02)
    assert result.fills.delta_qty.tolist() == pytest.approx([50, 3])


def test_risk_drift_is_trimmed_even_inside_large_trading_buffer():
    op, cl, target = market([[100, 100], [100, 100], [180, 100]],
                            weights=[[0.5, 0.5]] * 3)
    result = backtest.run_portfolio(op, cl, target, costs=FREE, initial_cash=10000,
                                    rebalance_band=0.3, max_asset=0.6)
    assert result.weights.iloc[-1].tolist() == pytest.approx([0.5, 0.5])
    assert result.fills.iloc[-1].reason == "risk_limit"


def test_future_data_does_not_change_historical_equity_or_fills():
    op, cl, target = market([[100], [102], [105], [104]], weights=[[0.5]] * 4)
    short = run(op.iloc[:3], cl.iloc[:3], target.iloc[:3])
    full = run(op, cl, target)
    pd.testing.assert_series_equal(short.equity, full.equity.iloc[:3])
    pd.testing.assert_frame_equal(short.fills, full.fills[full.fills.time <= op.index[2]].reset_index(drop=True))


@pytest.mark.parametrize("fault", ["nan_target", "leverage", "alignment", "missing_rate", "bad_price"])
def test_rejects_invalid_boundary_inputs(fault):
    op, cl, target = market([[100]] * 3, weights=[[0.5]] * 3)
    kwargs = {}
    if fault == "nan_target":
        target.iloc[0, 0] = np.nan
    elif fault == "leverage":
        target.iloc[0, 0] = 1.1
    elif fault == "alignment":
        target.index = target.index + pd.Timedelta(days=1)
    elif fault == "missing_rate":
        kwargs["borrow_rates"] = target.iloc[:2]
    else:
        op.iloc[1, 0] = 0
    with pytest.raises(ValueError):
        run(op, cl, target, **kwargs)


def test_insolvent_path_is_rejected_instead_of_reporting_valid_performance():
    op, cl, target = market([[100], [100]], [[100], [400]], [[-0.5], [-0.5]])
    with pytest.raises(RuntimeError, match="insolvent"):
        run(op, cl, target)

def test_idle_cash_earns_the_deposit_rate_while_positioned():
    op, cl, target = market([[100]] * 3, weights=[[0.5]] * 3)
    result = run(op, cl, target, costs=backtest.Costs(0, 0, 0, cash_annual=0.365, margin_annual=0.08))
    assert result.ledger.financing.tolist() == pytest.approx([0, 5.0, 5.005])
    assert result.equity.tolist() == pytest.approx([10000, 10005, 10010.005])


def test_negative_cash_pays_the_margin_rate():
    op, cl, target = market([[100]] * 3, weights=[[1.5]] * 3)
    result = backtest.run_portfolio(op, cl, target, initial_cash=10000, max_gross=2,
                                    max_asset=2, costs=backtest.Costs(0, 0, 0, margin_annual=0.365))
    assert result.ledger.financing.iloc[1] == pytest.approx(-5.0)
    assert result.equity.iloc[1] == pytest.approx(9995.0)
    assert result.ledger.gross_exposure.iloc[-1] == pytest.approx(
        15000 / result.equity.iloc[-1], rel=1e-9)
    assert result.ledger.gross_exposure.iloc[-1] > 1.5  # margin interest shrinks NAV


def test_leverage_above_one_requires_a_margin_rate():
    op, cl, target = market([[100]] * 2, weights=[[1.5]] * 2)
    with pytest.raises(ValueError, match="margin_annual"):
        backtest.run_portfolio(op, cl, target, max_gross=2, max_asset=2, costs=FREE)
