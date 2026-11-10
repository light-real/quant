# quant

BTC/ETH 日线趋势择时的**研究型**回测框架。不是实盘系统，不接交易接口。

结论一句话：在这段历史上，趋势择时的超额收益在验证段归零，它真正的作用是把最大回撤从 26% 压到 9%，代价是把收益从 13% 降到 6%。

## 结构

```
fetch_klines.py    从 Binance REST 镜像拉 5m/15m/1h/4h/1d/1w/1M，派生 1y，写 csv/ 与 manifest
verify_klines.py   独立审计：序列连续性、跨周期重聚合一致性
compare_sources.py 月度归档与 REST 镜像的双向差异（归档会漏掉迟到成交）

data.py            只保留已收线的 UTC 日线；拒绝缺口、重复、OHLC 违约
signals.py         趋势信号 = 20/60/120 日对数动量 ÷ 波动率 的裁剪均值
                   组合权重 = 逆波动率底仓 × 信号，15% 目标波动，总敞口 ≤1，单资产 ≤0.6
backtest.py        组合引擎：t 收盘出信号、t+1 开盘成交，成本不能融资出超额敞口
metrics.py         CAGR / 年化波动 / Sharpe / 最大回撤 / Calmar / 滚动 365 日
report.py          净值 + 回撤、实际敞口图
run_backtest.py    4 策略 × 开发/验证切分 × 风险匹配基准 × 成本情景
```

## 快速开始

```bash
pip install -r requirements.txt

# 1. 取数（约 450MB，写入 csv/，该目录不入库）
python fetch_klines.py --start 2018-11-01 --symbols BTCUSDT,ETHUSDT

# 2. 审计数据（可选但建议）
python verify_klines.py

# 3. 跑实验，结果写到 results_v2/<UTC时间戳>/
python run_backtest.py

# 4. 单元测试
python -m pytest -q
```

## 结果（验证段 2023-01-01 ~ 2026-11-07，1407 天）

| 策略 | CAGR | 年化波动 | Sharpe | 最大回撤 |
|---|---|---|---|---|
| passive_risk_budget（风险预算被动） | 12.96% | 15.98% | 0.84 | 26.34% |
| trend_long（只做多趋势） | 6.24% | 7.68% | 0.83 | 8.88% |
| trend_dual（多空趋势） | 3.36% | 9.86% | 0.38 | 12.74% |
| short_only（只做空，消融对照） | -2.65% | 6.33% | -0.39 | 14.49% |
| passive_matched_long（同波动率被动基准） | 7.36% | 8.90% | 0.84 | 15.50% |
| passive_matched_dual | 8.78% | 10.64% | 0.84 | 18.37% |

对照开发段（2019-03 ~ 2022-12）：趋势多头 Sharpe 1.67、回撤 6.88%，被动 Sharpe 1.01、回撤 26.61%。

读法：
- `trend_long` 的 Sharpe 0.83 与同波动率被动基准的 0.84 等价 —— **扣掉 4.68 倍年换手的手续费后，择时没有超额收益**。
- 空头是负贡献，`short_only` 在验证段净亏，且真实借券摩擦比模型更大。
- 2 倍交易成本下 `trend_dual` 的 Sharpe 从 0.38 掉到 0.25，对费用高度敏感。

## 已知局限

- 只用 BTC/ETH 两个高相关资产，组合实际上是单一 beta 的杠杆表达，没有做 universe 选择的修正。
- 验证段是**已经看过的历史**，不是样本外（`metadata.json` 里 `fresh_holdout: false`）。
- 空头为合成现货借券，未建模资金费率、强平、召回。
- 无杠杆清算、日内止损、市场冲击模型；闲置现金收益为零；期末持仓按市价标记而非平仓。
- 单笔成交的敞口上限可能被后续价格漂移突破。

## 环境

Python 3.12.7，pandas 3.0.3 / numpy 1.26.4 / matplotlib 3.9.2，311 个测试通过。
