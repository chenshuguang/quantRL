import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env_intraday_trend import IntradayTrendEnv


class TqdmProgressCallback(BaseCallback):
    """按训练步数实时刷新终端进度条。"""

    def __init__(self, total_steps: int):
        super().__init__()
        self.total_steps = total_steps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_steps, desc="Training", unit="step", dynamic_ncols=True)

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            remaining = self.total_steps - self.pbar.n
            if remaining > 0:
                self.pbar.update(remaining)
            self.pbar.close()


def load_1m_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "stime" in df.columns:
        dt = pd.to_datetime(df["stime"], format="%Y%m%d%H%M%S", errors="coerce")
    elif "time" in df.columns:
        dt = pd.to_datetime(df["time"], errors="coerce")
    else:
        raise ValueError("CSV 需要包含 stime 或 time 字段")

    df = df.assign(dt=dt).dropna(subset=["dt"]).set_index("dt").sort_index()

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要行情列: {missing}")

    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    return df


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.8):
    cut = int(len(df) * train_ratio)
    cut = max(500, min(cut, len(df) - 200))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def evaluate(model: PPO, df_test: pd.DataFrame, out_dir: Path):
    env = IntradayTrendEnv(df_test)
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    report = env.summary()
    out_dir.mkdir(parents=True, exist_ok=True)

    report["equity_df"].to_csv(out_dir / "equity_curve.csv", encoding="utf-8-sig")
    report["trades_df"].to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")

    print("\n=== Backtest Summary ===")
    print(f"Total Return: {report['total_return']:.2%}")
    print(f"Max Drawdown: {report['max_drawdown']:.2%}")
    print(f"Num Trades: {report['num_trades']}")


def main():
    parser = argparse.ArgumentParser(description="白银 1m 日内趋势交易 RL 训练")
    parser.add_argument("--csv", type=Path, required=True, help="1m K线 CSV 路径")
    parser.add_argument("--steps", type=int, default=200_000, help="训练总步数")
    parser.add_argument("--model-out", type=Path, default=Path("models/trend_intraday_ppo"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/trend_intraday"))
    args = parser.parse_args()

    df = load_1m_data(args.csv)
    train_df, test_df = split_train_test(df)

    env = DummyVecEnv([lambda: IntradayTrendEnv(train_df)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2e-4,
        gamma=0.995,
        n_steps=1024,
        batch_size=256,
        gae_lambda=0.95,
        ent_coef=1e-3,
        clip_range=0.2,
        verbose=0,
    )

    callback = TqdmProgressCallback(total_steps=args.steps)
    model.learn(total_timesteps=args.steps, callback=callback)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_out))

    evaluate(model, test_df, args.report_dir)
    print(f"\n模型已保存: {args.model_out}.zip")
    print(f"报告已保存: {args.report_dir}")


if __name__ == "__main__":
    main()
