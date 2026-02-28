import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window, min_periods=1).mean().bfill().ffill()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().sort_index()
    close = data["close"].astype(float)

    feat = pd.DataFrame(index=data.index)
    feat["open"] = data["open"].astype(float)
    feat["high"] = data["high"].astype(float)
    feat["low"] = data["low"].astype(float)
    feat["close"] = close
    feat["volume"] = data.get("volume", pd.Series(0.0, index=data.index)).astype(float)

    feat["ret_1"] = close.pct_change(1).fillna(0.0)
    feat["ret_5"] = close.pct_change(5).fillna(0.0)
    feat["ema_fast"] = close.ewm(span=12, adjust=False).mean()
    feat["ema_slow"] = close.ewm(span=48, adjust=False).mean()
    feat["trend_bias"] = (feat["ema_fast"] - feat["ema_slow"]) / (close + 1e-8)
    feat["atr14"] = compute_atr(feat, 14)
    feat["volatility"] = close.pct_change().rolling(30, min_periods=1).std().fillna(0.0)

    feat["hour_sin"] = np.sin(2 * np.pi * feat.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat.index.hour / 24)
    feat["minute_sin"] = np.sin(2 * np.pi * feat.index.minute / 60)
    feat["minute_cos"] = np.cos(2 * np.pi * feat.index.minute / 60)

    return feat.replace([np.inf, -np.inf], 0.0).fillna(0.0)


class IntradayTrendEnv(gym.Env):
    """一分钟趋势交易环境，仅做日内交易。"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: float = 1_000_000,
        contract_multiplier: float = 15.0,
        fee_rate: float = 0.00005,
        max_position: int = 1,
        atr_stop_loss: float = 1.5,
        trailing_stop_atr: float = 1.2,
        risk_penalty: float = 0.0001,
    ):
        super().__init__()

        if len(df) < 500:
            raise ValueError("数据量过少，至少需要 500 行分钟数据")

        self.raw_df = df.copy().sort_index()
        self.df = prepare_features(self.raw_df)

        self.initial_capital = float(initial_capital)
        self.contract_multiplier = float(contract_multiplier)
        self.fee_rate = float(fee_rate)
        self.max_position = int(max_position)
        self.atr_stop_loss = float(atr_stop_loss)
        self.trailing_stop_atr = float(trailing_stop_atr)
        self.risk_penalty = float(risk_penalty)

        self.feature_cols = list(self.df.columns)
        self.mean = self.df[self.feature_cols].mean()
        self.std = self.df[self.feature_cols].std().replace(0, 1e-6)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self.feature_cols) + 3,),
            dtype=np.float32,
        )

        self.reset(seed=None)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = 120

        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.trail_anchor = 0.0

        self.cash = self.initial_capital
        self.last_equity = self.initial_capital

        self.logs = []
        self.trades = []

        return self._obs(), {}

    def _obs(self):
        row = self.df.iloc[self.ptr][self.feature_cols]
        normalized = ((row - self.mean) / (self.std + 1e-8)).to_numpy(dtype=np.float32)
        account = np.array(
            [
                float(self.position),
                (self.entry_price - row["close"]) / (row["close"] + 1e-8),
                (self.stop_price - row["close"]) / (row["close"] + 1e-8),
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(np.concatenate([normalized, account]), nan=0.0, posinf=10.0, neginf=-10.0)

    def _fee(self, price: float, size: int = 1) -> float:
        return abs(size) * price * self.contract_multiplier * self.fee_rate

    def _unrealized(self, price: float) -> float:
        if self.position == 0:
            return 0.0
        diff = price - self.entry_price
        return diff * self.contract_multiplier * self.position

    def _close(self, price: float, reason: str):
        if self.position == 0:
            return 0.0

        pnl = self._unrealized(price)
        fee = self._fee(price, abs(self.position))
        self.cash += pnl - fee
        self.trades.append(
            {
                "time": self.df.index[self.ptr],
                "side": "LONG" if self.position > 0 else "SHORT",
                "entry": self.entry_price,
                "exit": price,
                "pnl": pnl - fee,
                "reason": reason,
            }
        )

        self.position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.trail_anchor = 0.0
        return pnl - fee

    def _open(self, direction: int, price: float):
        direction = int(np.sign(direction))
        if direction == 0:
            return
        if self.position != 0:
            return

        self.position = direction * self.max_position
        self.entry_price = price
        atr = max(1e-6, float(self.df["atr14"].iloc[self.ptr]))

        if self.position > 0:
            self.stop_price = price - self.atr_stop_loss * atr
            self.trail_anchor = price
        else:
            self.stop_price = price + self.atr_stop_loss * atr
            self.trail_anchor = price

        self.cash -= self._fee(price, abs(self.position))

    def _update_trailing_stop(self, price: float):
        if self.position == 0:
            return

        atr = max(1e-6, float(self.df["atr14"].iloc[self.ptr]))
        if self.position > 0:
            self.trail_anchor = max(self.trail_anchor, price)
            trailing = self.trail_anchor - self.trailing_stop_atr * atr
            self.stop_price = max(self.stop_price, trailing)
        else:
            self.trail_anchor = min(self.trail_anchor, price)
            trailing = self.trail_anchor + self.trailing_stop_atr * atr
            self.stop_price = min(self.stop_price, trailing)

    def _force_intraday_flat(self):
        now = self.df.index[self.ptr]
        if self.ptr + 1 >= len(self.df):
            return True
        nxt = self.df.index[self.ptr + 1]
        return now.date() != nxt.date()

    def step(self, action: int):
        price = float(self.df["close"].iloc[self.ptr])
        terminated = False
        truncated = False

        # 先更新移动止损
        self._update_trailing_stop(price)

        # 止损触发
        if self.position > 0 and price <= self.stop_price:
            self._close(price, reason="stop_loss_or_trailing")
        elif self.position < 0 and price >= self.stop_price:
            self._close(price, reason="stop_loss_or_trailing")

        # 策略动作
        if action == 1:
            if self.position < 0:
                self._close(price, reason="reverse_to_long")
            self._open(direction=1, price=price)
        elif action == 2:
            if self.position > 0:
                self._close(price, reason="reverse_to_short")
            self._open(direction=-1, price=price)
        elif action == 3:
            self._close(price, reason="manual_flat")

        # 日内强平
        if self.position != 0 and self._force_intraday_flat():
            self._close(price, reason="end_of_day_flat")

        equity = self.cash + self._unrealized(price)
        reward = (equity - self.last_equity) / self.initial_capital
        reward -= self.risk_penalty * abs(self.position)

        self.logs.append(
            {
                "time": self.df.index[self.ptr],
                "price": price,
                "position": self.position,
                "stop_price": self.stop_price,
                "equity": equity,
                "reward": reward,
            }
        )

        self.last_equity = equity
        self.ptr += 1

        if self.ptr >= len(self.df) - 1:
            terminated = True

        return self._obs(), float(reward), terminated, truncated, {
            "equity": equity,
            "position": self.position,
        }

    def summary(self):
        if not self.logs:
            return {}
        log_df = pd.DataFrame(self.logs).set_index("time")
        trade_df = pd.DataFrame(self.trades)

        total_return = (log_df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        drawdown = (log_df["equity"] - log_df["equity"].cummax()) / log_df["equity"].cummax()
        max_dd = drawdown.min()

        return {
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "num_trades": int(len(trade_df)),
            "equity_df": log_df,
            "trades_df": trade_df,
        }
