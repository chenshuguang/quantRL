import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import math
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 辅助函数：特征工程
# ============================================================

def filter_trading_sessions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "volume" in df.columns:
        df = df[df["volume"] > 0]
    return df.sort_index()

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev).abs()
    tr3 = (low - prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=1).mean()
    return atr.fillna(method="bfill").fillna(method="ffill")

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"].astype(float)

    df_feat = pd.DataFrame(index=df.index)
    df_feat["close"] = close
    df_feat["open"] = df["open"].astype(float)
    df_feat["high"] = df["high"].astype(float)
    df_feat["low"] = df["low"].astype(float)
    df_feat["volume"] = df.get("volume", pd.Series(0, index=df.index)).astype(float)

    df_feat["ma20"] = close.rolling(20, min_periods=1).mean()
    df_feat["ma60"] = close.rolling(60, min_periods=1).mean()
    df_feat["mom5"] = close.pct_change(5).fillna(0)
    df_feat["mom20"] = close.pct_change(20).fillna(0)
    df_feat["atr14"] = compute_atr(df, 14)
    df_feat["rsi14"] = compute_rsi(close, 14)
    df_feat["vol30"] = close.pct_change().rolling(30, min_periods=1).std().fillna(0)
    df_feat["trend_strength"] = (df_feat["ma20"] - df_feat["ma60"]).abs() / (df_feat["atr14"] + 1e-8)
    df_feat["trend_direction"] = np.sign(df_feat["ma20"] - df_feat["ma60"])
    df_feat["chop_ratio"] = (df_feat["vol30"] / (df_feat["mom20"].abs() + 1e-8)).clip(0, 10)

    hours = df.index.hour
    minutes = df.index.minute
    df_feat["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df_feat["minute_sin"] = np.sin(2 * np.pi * minutes / 60)
    df_feat["minute_cos"] = np.cos(2 * np.pi * minutes / 60)

    df_feat = df_feat.replace([np.inf, -np.inf], 0).fillna(method="bfill").fillna(0)
    return df_feat

# ============================================================
# 专业版环境（Gymnasium API）
# ============================================================

class SilverTrendRLEnv(gym.Env):
    """
    专业版 StopRun 环境（100% Gymnasium API）
    - reset() → (obs, info)
    - step() → (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1_000_000,
        contract_multiplier: float = 15.0,
        fee_rate: float = 0.00005 * 2.0,
        max_position: int = 3,
        atr_stop_multiplier: float = 2.0,
        atr_tp_multiplier: float = 4.0,
        min_hold_bars: int = 5,
        early_close_minutes: int = 10,
        trade_penalty: float = 0.0,
        enable_time_filter: bool = True,
        trend_filter_strength: float = 0.25,
        max_chop_ratio: float = 2.5,
        cooldown_bars: int = 2,
        drawdown_penalty: float = 0.0,
    ):
        super().__init__()

        df = filter_trading_sessions(df)
        self.raw_df = df.copy()
        self.df = prepare_features(df)

        self.initial_capital = initial_capital
        self.contract_multiplier = contract_multiplier
        self.fee_rate = fee_rate
        self.max_position = max_position

        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_tp_multiplier = atr_tp_multiplier
        self.min_hold_bars = min_hold_bars
        self.early_close_minutes = early_close_minutes
        self.trade_penalty = trade_penalty
        self.enable_time_filter = enable_time_filter
        self.trend_filter_strength = trend_filter_strength
        self.max_chop_ratio = max_chop_ratio
        self.cooldown_bars = cooldown_bars
        self.drawdown_penalty = drawdown_penalty

        self.obs_cols = list(self.df.columns)
        self.obs_mean = self.df[self.obs_cols].mean()
        self.obs_std = self.df[self.obs_cols].std() + 1e-8

        # 动作空间（9 动作）
        self.action_space = spaces.Discrete(9)

        # 观测空间：特征 + position
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(len(self.obs_cols) + 1,), dtype=np.float32
        )

        # 初始化
        self.reset(seed=None)

    # ============================================================
    # Gymnasium reset()
    # ============================================================
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = max(200, int(len(self.df) * 0.01))
        if self.t >= len(self.df) - 2:
            self.t = 1

        self.position = 0
        self.entry_price = None
        self.entry_time = None
        self.entry_size = 0

        self.cash = float(self.initial_capital)
        self.last_equity = float(self.initial_capital)
        self.last_price = float(self.df["close"].iloc[self.t])

        self.total_fee = 0.0
        self.trade_log = []
        self.equity_curve = []
        self.hold_bars = 0
        self.cooldown_remaining = 0
        self.peak_equity = float(self.initial_capital)

        obs = self._get_state()
        info = {}
        return obs, info

    # ============================================================
    # 工具函数
    # ============================================================

    def _get_state(self):
        row = self.df.iloc[self.t][self.obs_cols]
        norm = (row - self.obs_mean) / self.obs_std
        obs = np.concatenate([norm.values.astype(np.float32), [float(self.position)]])
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _contract_notional(self, price: float, size: int):
        return float(price) * self.contract_multiplier * abs(size)

    def _calculate_fee(self, price: float, size: int):
        return self._contract_notional(price, size) * self.fee_rate

    def _unrealized_pnl(self, price: float):
        if self.position == 0 or self.entry_price is None:
            return 0.0
        diff = float(price) - float(self.entry_price)
        if self.position > 0:
            return diff * self.contract_multiplier * self.entry_size
        else:
            return -diff * self.contract_multiplier * self.entry_size

    def _current_atr(self):
        return float(self.df["atr14"].iloc[self.t])

    def _is_early_close_time(self, current_time):
        h = current_time.hour
        m = current_time.minute
        if (h == 14 and m >= 60 - self.early_close_minutes) or \
           (h == 2 and m >= 30 - self.early_close_minutes):
            return True
        return False

    def _can_open_new_position(self, direction: int):
        row = self.df.iloc[self.t]
        trend_strength = float(row["trend_strength"])
        trend_direction = int(np.sign(row["trend_direction"]))
        chop_ratio = float(row["chop_ratio"])

        if trend_strength < self.trend_filter_strength:
            return False
        if chop_ratio > self.max_chop_ratio:
            return False
        if trend_direction != 0 and trend_direction != direction:
            return False
        return True

    # ============================================================
    # 加仓 / 减仓 / 平仓逻辑
    # ============================================================

    def _open_position(self, direction: int, price: float, size: int = 1):
        realized = 0.0
        fee = 0.0

        # 如果方向相反 → 先平仓
        if self.position != 0 and np.sign(self.position) != direction:
            realized += self._close_all(price)
            fee += self._calculate_fee(price, self.entry_size)

        # 新仓位大小
        new_size = abs(self.position) + size if self.position != 0 else size
        new_size = min(new_size, self.max_position)

        # 更新仓位
        if self.position == 0:
            self.entry_price = price
        else:
            # 加仓时重新计算均价
            self.entry_price = (
                self.entry_price * self.entry_size + price * size
            ) / (self.entry_size + size)

        self.position = direction * new_size
        self.entry_size = new_size
        self.entry_time = self.df.index[self.t]

        fee += self._calculate_fee(price, size)
        return realized, fee

    def _reduce_position(self, reduce_size: int, price: float):
        if self.position == 0:
            return 0.0, 0.0

        reduce_size = min(abs(self.position), reduce_size)
        realized = 0.0
        fee = 0.0

        if self.position > 0:
            pnl = (price - self.entry_price) * self.contract_multiplier * reduce_size
        else:
            pnl = (self.entry_price - price) * self.contract_multiplier * reduce_size

        realized += pnl
        fee += self._calculate_fee(price, reduce_size)

        remaining = abs(self.position) - reduce_size
        if remaining == 0:
            self.position = 0
            self.entry_price = None
            self.entry_size = 0
            self.entry_time = None
        else:
            self.position = np.sign(self.position) * remaining
            self.entry_size = remaining

        self.cash += realized
        return realized, fee

    def _close_all(self, price: float):
        if self.position == 0 or self.entry_price is None:
            return 0.0

        if self.position > 0:
            pnl = (price - self.entry_price) * self.contract_multiplier * self.entry_size
        else:
            pnl = (self.entry_price - price) * self.contract_multiplier * self.entry_size

        self.cash += pnl
        self.position = 0
        self.entry_price = None
        self.entry_size = 0
        self.entry_time = None
        return pnl

    # ============================================================
    # Gymnasium step()
    # ============================================================

    def step(self, action: int):
        current_price = float(self.df["close"].iloc[self.t])
        current_time = self.df.index[self.t]

        realized = 0.0
        fee = 0.0
        trade_executed = False

        # 持仓 bar 计数
        if self.position != 0:
            self.hold_bars += 1
        else:
            self.hold_bars = 0

        # 提前强制平仓
        if self.enable_time_filter and self._is_early_close_time(current_time):
            if self.position != 0 and self.hold_bars >= self.min_hold_bars:
                close_size = self.entry_size
                realized += self._close_all(current_price)
                fee += self._calculate_fee(current_price, close_size)
                trade_executed = True
                self.cooldown_remaining = self.cooldown_bars
            # 禁止开仓
            if action in [1, 2, 5, 6]:
                action = 0

        # ATR 止损 / 止盈
        if self.position != 0:
            atr = max(1e-6, self._current_atr())
            stop_points = self.atr_stop_multiplier * atr
            tp_points = self.atr_tp_multiplier * atr

            unreal = self._unrealized_pnl(current_price)
            unreal_points = unreal / self.contract_multiplier

            # 止损
            if unreal_points <= -stop_points:
                close_size = self.entry_size
                realized += self._close_all(current_price)
                fee += self._calculate_fee(current_price, close_size)
                trade_executed = True
                self.cooldown_remaining = self.cooldown_bars

            # 止盈
            elif unreal_points >= tp_points:
                close_size = self.entry_size
                realized += self._close_all(current_price)
                fee += self._calculate_fee(current_price, close_size)
                trade_executed = True
                self.cooldown_remaining = self.cooldown_bars

        # 冷静期：禁止开新仓，避免高频反复打止损
        if self.cooldown_remaining > 0 and action in [1, 2, 5, 6]:
            action = 0

        # ============================================================
        # 动作执行（9 动作）
        # ============================================================

        if action == 1:  # 开多
            if self._can_open_new_position(1):
                r, f = self._open_position(1, current_price, size=1)
                realized += r; fee += f; trade_executed = True

        elif action == 2:  # 加多
            if self.position > 0:
                r, f = self._open_position(1, current_price, size=1)
                realized += r; fee += f; trade_executed = True

        elif action == 3:  # 减多
            if self.position > 0:
                r, f = self._reduce_position(1, current_price)
                realized += r; fee += f; trade_executed = True

        elif action == 4:  # 平多
            if self.position > 0:
                close_size = self.entry_size
                r = self._close_all(current_price)
                fee += self._calculate_fee(current_price, close_size)
                realized += r; trade_executed = True
                self.cooldown_remaining = self.cooldown_bars

        elif action == 5:  # 开空
            if self._can_open_new_position(-1):
                r, f = self._open_position(-1, current_price, size=1)
                realized += r; fee += f; trade_executed = True

        elif action == 6:  # 加空
            if self.position < 0:
                r, f = self._open_position(-1, current_price, size=1)
                realized += r; fee += f; trade_executed = True

        elif action == 7:  # 减空
            if self.position < 0:
                r, f = self._reduce_position(1, current_price)
                realized += r; fee += f; trade_executed = True

        elif action == 8:  # 平空
            if self.position < 0:
                close_size = self.entry_size
                r = self._close_all(current_price)
                fee += self._calculate_fee(current_price, close_size)
                realized += r; trade_executed = True
                self.cooldown_remaining = self.cooldown_bars

        # 手续费真实计入现金
        if fee > 0:
            self.cash -= fee
            self.total_fee += fee

        # ============================================================
        # 计算奖励（Δequity）
        # ============================================================

        unrealized = self._unrealized_pnl(current_price)
        total_equity = self.cash + unrealized

        reward = total_equity - self.last_equity
        if trade_executed:
            reward -= self.trade_penalty

        self.peak_equity = max(self.peak_equity, total_equity)
        current_drawdown = (self.peak_equity - total_equity) / max(1e-6, self.peak_equity)
        reward -= self.drawdown_penalty * current_drawdown

        # 记录 equity
        self.equity_curve.append({
            "time": current_time,
            "price": current_price,
            "position": int(self.position),
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "cash": float(self.cash),
            "unrealized_pnl": float(unrealized),
            "total_equity": float(total_equity),
            "realized": float(realized),
            "fee": float(fee),
            "reward": float(reward),
        })

        # 记录交易
        if trade_executed:
            self.trade_log.append({
                "time": current_time,
                "action": int(action),
                "price": current_price,
                "position_after": int(self.position),
                "realized": float(realized),
                "fee": float(fee),
                "cash": float(self.cash),
                "total_equity": float(total_equity),
            })

        # 前进
        self.last_equity = total_equity
        self.last_price = current_price
        self.t += 1
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        terminated = (self.t >= len(self.df) - 1)
        truncated = False

        obs = self._get_state()
        info = {}

        return obs, float(reward), terminated, truncated, info

    # ============================================================
    # 回测指标
    # ============================================================

    def get_backtest_metrics(self):
        if not self.equity_curve:
            return {}

        equity_df = pd.DataFrame(self.equity_curve).set_index("time")
        final_equity = equity_df["total_equity"].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        cummax = equity_df["total_equity"].cummax()
        drawdown = (equity_df["total_equity"] - cummax) / cummax
        max_dd = drawdown.min()

        returns = equity_df["total_equity"].pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = np.sqrt(252 * 240) * returns.mean() / returns.std()
        else:
            sharpe = 0.0

        trades_df = pd.DataFrame(self.trade_log)

        return {
            "final_equity": float(final_equity),
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "sharpe_ratio": float(sharpe),
            "total_trades": len(trades_df),
            "trades_df": trades_df,
            "equity_df": equity_df,
        }

    # ============================================================
    # render（可选）
    # ============================================================

    def render(self):
        if len(self.equity_curve) == 0:
            print("No data")
            return
        print(pd.DataFrame(self.equity_curve).tail(5))
