from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    fee: float = 0.001
    slippage: float = 0.0005
    borrow_annual: float = 0.10
    # Interest rates on the cash balance itself, both annual fractions.
    cash_annual: float = 0.0    # positive cash earns this, e.g. a money-market rate
    margin_annual: float = 0.0  # negative cash (debit) pays this, e.g. a margin loan


@dataclass
class PortfolioResult:
    equity: pd.Series
    weights: pd.DataFrame
    fills: pd.DataFrame
    ledger: pd.DataFrame
    asset_pnl: pd.DataFrame


FILL_COLUMNS = ["time", "symbol", "delta_qty", "price", "notional", "fee", "slippage", "reason"]
LEDGER_COLUMNS = ["cash", "fees", "slippage", "borrow", "turnover", "long_pnl",
                  "short_pnl", "gross_exposure", "net_exposure", "n_fills", "financing"]


def _validate_frames(opens, closes, targets, borrow_rates):
    idx = opens.index
    if (opens.empty or not isinstance(idx, pd.DatetimeIndex) or str(idx.tz) != "UTC"
            or not idx.is_unique or not idx.is_monotonic_increasing
            or not idx.equals(idx.normalize()) or not opens.columns.is_unique):
        raise ValueError("prices require a unique increasing UTC daily index and columns")
    if len(idx) > 1 and not (idx.to_series().diff().iloc[1:] == pd.Timedelta(days=1)).all():
        raise ValueError("gap in execution prices")
    frames = [opens, closes, targets] + ([borrow_rates] if borrow_rates is not None else [])
    for frame in frames:
        if not frame.index.equals(idx) or not frame.columns.equals(opens.columns):
            raise ValueError("prices, targets and borrow rates must align exactly")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("prices, targets and borrow rates must be finite")
    if (opens <= 0).any().any() or (closes <= 0).any().any():
        raise ValueError("prices must be positive")
    if borrow_rates is not None and (borrow_rates < 0).any().any():
        raise ValueError("borrow rates must be nonnegative annual rates")


def _rebalance_quantity(qty, price, target, equity, costs):
    # Solve weights against post-cost NAV so fees cannot finance extra exposure.
    low, high = 0.0, equity
    for _ in range(50):
        nav = (low + high) / 2
        candidate = target * nav / price
        delta = candidate - qty
        fills = price * (1 + np.sign(delta) * costs.slippage)
        cost = np.sum(np.abs(delta) * (price * costs.slippage + fills * costs.fee))
        if nav + cost > equity:
            high = nav
        else:
            low = nav
    return target * ((low + high) / 2) / price


def run_portfolio(opens: pd.DataFrame, closes: pd.DataFrame, targets: pd.DataFrame,
                  costs: Costs = Costs(), initial_cash: float = 20000.0,
                  rebalance_band: float = 0.02, max_gross: float = 1.0,
                  max_asset: float = 0.6,
                  borrow_rates: pd.DataFrame | None = None) -> PortfolioResult:
    _validate_frames(opens, closes, targets, borrow_rates)
    params = [initial_cash, rebalance_band, max_gross, max_asset,
              costs.fee, costs.slippage, costs.borrow_annual,
              costs.cash_annual, costs.margin_annual]
    if (not np.isfinite(params).all() or initial_cash <= 0 or not 0 <= rebalance_band <= 1
            or max_gross <= 0 or max_asset <= 0
            or not 0 <= costs.fee < 0.1 or not 0 <= costs.slippage < 0.1
            or costs.borrow_annual < 0
            or not 0 <= costs.cash_annual < 1 or not 0 <= costs.margin_annual < 1):
        raise ValueError("invalid capital, cost or exposure parameters")
    if (max_gross > 1 + 1e-10 or max_asset > 1 + 1e-10) and costs.margin_annual <= 0:
        raise ValueError("leverage above 1x requires a positive margin_annual rate")
    if ((targets.abs().sum(axis=1) > max_gross + 1e-10).any()
            or (targets.abs() > max_asset + 1e-10).any().any()):
        raise ValueError("target exceeds exposure limit")

    o, c, target = (frame.to_numpy(dtype=float) for frame in (opens, closes, targets))
    rates = (borrow_rates.to_numpy(dtype=float) if borrow_rates is not None
             else np.full(o.shape, costs.borrow_annual))
    n, assets = o.shape
    cash, qty = float(initial_cash), np.zeros(assets)
    equity = np.full(n, initial_cash, dtype=float)
    realized_weights = np.zeros_like(o)
    asset_pnl = np.zeros_like(o)
    ledger = np.zeros((n, len(LEDGER_COLUMNS)))
    ledger[0, 0] = initial_cash
    fills = []

    for t in range(1, n):
        old_qty = qty.copy()
        gap = old_qty * (o[t] - c[t - 1])
        open_nav = cash + np.dot(qty, o[t])
        if open_nav <= 0:
            raise RuntimeError(f"insolvent at {opens.index[t]} open")
        current = qty * o[t] / open_nav
        desired = target[t - 1]
        risk_limit = np.abs(current).sum() > max_gross + 1e-10 or (np.abs(current) > max_asset + 1e-10).any()
        direction_change = (np.sign(current) != np.sign(desired)).any()
        distance = np.abs(desired - current).max()
        trade = risk_limit or direction_change or distance > max(rebalance_band, 1e-10)
        fee = np.zeros(assets)
        slip = np.zeros(assets)
        long_cost = short_cost = turnover = 0.0
        count = 0

        if trade:
            qty = _rebalance_quantity(old_qty, o[t], desired, open_nav, costs)
            delta = qty - old_qty
            fill_prices = o[t] * (1 + np.sign(delta) * costs.slippage)
            fee = np.abs(delta) * fill_prices * costs.fee
            slip = np.abs(delta) * o[t] * costs.slippage
            cash -= np.dot(delta, fill_prices) + fee.sum()
            turnover = np.dot(np.abs(delta), o[t]) / open_nav
            for j in range(assets):
                if abs(delta[j]) * o[t, j] < 1e-8:
                    continue
                count += 1
                closed = min(abs(old_qty[j]), abs(delta[j])) if old_qty[j] * delta[j] < 0 else 0.0
                long_units = closed if old_qty[j] > 0 else 0.0
                long_units += abs(delta[j]) - closed if delta[j] > 0 else 0.0
                fraction = long_units / abs(delta[j])
                total_cost = fee[j] + slip[j]
                long_cost += total_cost * fraction
                short_cost += total_cost * (1 - fraction)
                fills.append({"time": opens.index[t], "symbol": opens.columns[j],
                              "delta_qty": delta[j], "price": fill_prices[j],
                              "notional": abs(delta[j]) * fill_prices[j], "fee": fee[j],
                              "slippage": slip[j], "reason": "risk_limit" if risk_limit else "rebalance"})

        intraday = qty * (c[t] - o[t])
        borrow = np.maximum(-qty, 0) * c[t] * rates[t] / 365
        cash -= borrow.sum()
        # Idle cash earns the deposit rate; a debit balance pays the margin rate.
        financing = cash * (costs.cash_annual if cash >= 0 else costs.margin_annual) / 365
        cash += financing
        equity[t] = cash + np.dot(qty, c[t])
        if equity[t] <= 0:
            raise RuntimeError(f"insolvent at {opens.index[t]} close")
        asset_pnl[t] = gap + intraday - fee - slip - borrow
        long_pnl = gap[old_qty > 0].sum() + intraday[qty > 0].sum() - long_cost
        short_pnl = gap[old_qty < 0].sum() + intraday[qty < 0].sum() - short_cost - borrow.sum()
        realized_weights[t] = qty * c[t] / equity[t]
        ledger[t] = [cash, fee.sum(), slip.sum(), borrow.sum(), turnover, long_pnl,
                     short_pnl, np.abs(realized_weights[t]).sum(), realized_weights[t].sum(), count,
                     financing]

    return PortfolioResult(
        equity=pd.Series(equity, index=opens.index, name="equity"),
        weights=pd.DataFrame(realized_weights, index=opens.index, columns=opens.columns),
        fills=pd.DataFrame(fills, columns=FILL_COLUMNS),
        ledger=pd.DataFrame(ledger, index=opens.index, columns=LEDGER_COLUMNS),
        asset_pnl=pd.DataFrame(asset_pnl, index=opens.index, columns=opens.columns),
    )
