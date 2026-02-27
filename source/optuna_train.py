# optuna_train.py —— Gymnasium 版本（与 StopRunPro 环境完全匹配）
import os
import sys
import logging
from pathlib import Path
import optuna
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 你的专业版环境（Gymnasium API）
from env_silver_trend_rl import SilverTrendRLEnv

# 分析与绘图
from analyze_trades import analyze_trades
from plot_trades import plot_trades, load_data as load_price_data


# ============================================================
# 路径配置
# ============================================================

CSV_PATH = r"D:\work\auto_quant\csv\silver_1m.csv"
MODEL_DIR = Path("./models")
LOG_DIR = Path(r"D:\work\auto_quant\csv")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 日志
# ============================================================

logger = logging.getLogger("optuna_stoprun_pro")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_DIR / "optuna_train_stoprun_pro.log", encoding="utf-8")
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
fh.setFormatter(fmt)
logger.addHandler(fh)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)


# ============================================================
# 数据加载
# ============================================================

_cached_df = None

def load_df():
    global _cached_df
    if _cached_df is not None:
        return _cached_df.copy()

    df = pd.read_csv(CSV_PATH)
    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    if "stime" in df.columns:
        df["dt"] = pd.to_datetime(df["stime"], format="%Y%m%d%H%M%S", errors="coerce")
    elif "time" in df.columns:
        try:
            df["dt"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
        except Exception:
            df["dt"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        raise ValueError("CSV 必须包含 stime 或 time 列")

    df = df.set_index("dt").sort_index()
    _cached_df = df

    logger.info(f"数据加载完成: {len(df)} 条, 时间范围: {df.index[0]} ~ {df.index[-1]}")
    return df.copy()


# ============================================================
# 环境工厂（Gymnasium）
# ============================================================

def make_env_fn(df, params):
    def _init():
        return SilverTrendRLEnv(
            df,
            initial_capital=1_000_000,
            contract_multiplier=15.0,
            fee_rate=0.00005 * 2.0,
            max_position=3,
            atr_stop_multiplier=params["atr_stop_multiplier"],
            atr_tp_multiplier=params["atr_tp_multiplier"],
            min_hold_bars=params["min_hold_bars"],
            early_close_minutes=10,
            trade_penalty=params["trade_penalty"],
            enable_time_filter=True,
        )
    return _init


# ============================================================
# 回测（Gymnasium API）
# ============================================================

def run_backtest_from_model(model, df, params):
    env = SilverTrendRLEnv(
        df,
        initial_capital=1_000_000,
        contract_multiplier=15.0,
        fee_rate=0.00005 * 2.0,
        max_position=3,
        atr_stop_multiplier=params["atr_stop_multiplier"],
        atr_tp_multiplier=params["atr_tp_multiplier"],
        min_hold_bars=params["min_hold_bars"],
        early_close_minutes=10,
        trade_penalty=params["trade_penalty"],
        enable_time_filter=True,
    )

    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break

    return env.get_backtest_metrics()


# ============================================================
# Optuna objective
# ============================================================

def objective(trial: optuna.Trial):
    df = load_df()

    params = {
        "atr_stop_multiplier": trial.suggest_float("atr_stop_multiplier", 1.0, 4.0),
        "atr_tp_multiplier": trial.suggest_float("atr_tp_multiplier", 2.0, 6.0),
        "min_hold_bars": trial.suggest_int("min_hold_bars", 1, 30),
        "trade_penalty": trial.suggest_float("trade_penalty", 0.0, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-6, 0.01, log=True),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999),
    }

    env = DummyVecEnv([make_env_fn(df, params)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=params["learning_rate"],
        n_steps=2048,
        batch_size=512,
        gamma=params["gamma"],
        gae_lambda=0.95,
        ent_coef=params["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    short_steps = 200_000
    logger.info(f"Trial {trial.number} 开始短训 {short_steps} steps, params={params}")
    model.learn(total_timesteps=short_steps)

    metrics = run_backtest_from_model(model, df, params)

    # 保存 trial 结果
    eq = metrics.get("equity_df")
    tr = metrics.get("trades_df")

    eq_path = LOG_DIR / f"optuna_trial_{trial.number:03d}_equity.csv"
    tr_path = LOG_DIR / f"optuna_trial_{trial.number:03d}_trades.csv"

    if eq is not None:
        eq.to_csv(eq_path, encoding="utf-8-sig")
    if tr is not None:
        tr.to_csv(tr_path, index=False, encoding="utf-8-sig")

    sharpe = metrics.get("sharpe_ratio", 0.0)
    total_trades = metrics.get("total_trades", 0)

    objective_value = sharpe - 0.0002 * max(0, total_trades - 5000)
    return objective_value


# ============================================================
# 主流程：Optuna + 长训 + 回测 + 分析 + 绘图
# ============================================================

def train_best(n_trials: int = 8):
    df = load_df()

    study = optuna.create_study(direction="maximize")
    logger.info("开始 Optuna 调参")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    params = best.params

    logger.info("===== 最优参数 =====")
    for k, v in params.items():
        logger.info(f"{k}: {v}")

    # 长训
    env = DummyVecEnv([make_env_fn(df, params)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=params["learning_rate"],
        n_steps=4096,
        batch_size=512,
        gamma=params["gamma"],
        gae_lambda=0.95,
        ent_coef=params["ent_coef"],
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    long_steps = 500_000
    logger.info(f"开始长训 {long_steps} steps")
    model.learn(total_timesteps=long_steps)

    # 保存模型
    best_model = MODEL_DIR / "ppo_silver_stoprun_pro_best.zip"
    model.save(best_model)
    logger.info(f"模型已保存: {best_model}")

    # 保存标准化参数
    env0 = env.envs[0]
    stats_path = MODEL_DIR / "ppo_silver_stoprun_pro_best_stats.npz"
    np.savez(stats_path, mean=env0.obs_mean.values, std=env0.obs_std.values)
    logger.info(f"标准化参数已保存: {stats_path}")

    # 正式回测
    metrics = run_backtest_from_model(model, df, params)

    equity_df = metrics["equity_df"]
    trades_df = metrics["trades_df"]

    equity_path = LOG_DIR / "rl_equity_curve_stoprun_pro.csv"
    trades_path = LOG_DIR / "rl_trades_completed_stoprun_pro.csv"

    equity_df.to_csv(equity_path)
    trades_df.to_csv(trades_path, index=False)

    logger.info("回测结果已保存")

    # 自动分析
    analyze_trades(str(trades_path))

    # 自动绘图
    price_df = load_price_data(CSV_PATH)
    plot_trades(price_df, str(trades_path))


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    train_best(n_trials=8)
