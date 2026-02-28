# trend_intraday_project

新的白银 1 分钟量化训练项目（RL 版），核心目标：

- **趋势交易**：使用 EMA 快慢线与动量特征驱动策略学习。
- **仅日内交易**：跨日时自动强制平仓，不留隔夜仓位。
- **止损机制**：同时使用 ATR 固定止损 + ATR 移动止损（trailing stop）。
- **训练进度条**：训练时在终端按 step 实时刷新 `tqdm` 进度条。

---

## 目录

- `env_intraday_trend.py`：环境定义与风控逻辑。
- `train_trend_intraday.py`：训练入口、进度条回调、回测与结果导出。

---

## 依赖

```bash
pip install gymnasium stable-baselines3 pandas numpy tqdm
```

---

## 运行方式

```bash
python source/trend_intraday_project/train_trend_intraday.py \
  --csv /path/to/silver_1m.csv \
  --steps 200000 \
  --model-out models/trend_intraday_ppo \
  --report-dir reports/trend_intraday
```

> CSV 至少需要列：`open, high, low, close`，并包含 `stime`（`%Y%m%d%H%M%S`）或 `time` 时间列。

---

## 动作定义

- `0`: 保持
- `1`: 做多（若空仓则开多，若有空单先平空）
- `2`: 做空（若空仓则开空，若有多单先平多）
- `3`: 平仓

---

## 风控细节

1. **固定止损**：开仓时按照 `atr_stop_loss * ATR` 设定初始止损位。
2. **移动止损**：持仓后根据最新有利价格更新 `trail_anchor`，按 `trailing_stop_atr * ATR` 抬升/下移止损。
3. **日内强平**：当下一根 K 线进入新交易日时，本 bar 自动平仓。

---

## 输出结果

训练完成后输出：

- 模型：`<model-out>.zip`
- 回测净值曲线：`<report-dir>/equity_curve.csv`
- 交易明细：`<report-dir>/trades.csv`

终端会打印回测摘要（收益率、最大回撤、交易次数）。
