"""
特征工程模块 — ETF LSTM 预测

从 ETF 日线数据中计算技术指标特征。

注意：baostock 数据约 96 行（2026-01 起），滚动窗口控制在 20 天内。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from config.etf_config import (
    MA_PERIODS, RSI_PERIODS,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLL_PERIOD, BOLL_STD, WINDOW_SIZE,
)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """对单只 ETF 计算全部特征。"""
    data = df.copy().sort_values('date').reset_index(drop=True)

    # ========== 基础特征 ==========
    data['ret_1d'] = data['close'].pct_change() * 100
    data['ret_5d'] = data['close'].pct_change(5) * 100
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1)) * 100
    data['amplitude'] = (data['high'] - data['low']) / data['close'].shift(1) * 100

    # ========== 移动平均线 ==========
    for p in MA_PERIODS:  # [5, 10, 20]
        ma = data['close'].rolling(p).mean()
        data[f'MA{p}'] = ma
        data[f'MA{p}_dev'] = (data['close'] - ma) / ma * 100

    # ========== RSI ==========
    for p in RSI_PERIODS:  # [6, 12]
        data[f'RSI{p}'] = _rsi(data['close'], p)

    # ========== MACD ==========
    dif, dea, hist = _macd(data['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    data['MACD_DIF'] = dif
    data['MACD_DEA'] = dea
    data['MACD_HIST'] = hist

    # ========== 布林带 ==========
    mid, upper, lower, width, pos = _bollinger(data['close'], BOLL_PERIOD, BOLL_STD)
    data['BB_mid'] = mid
    data['BB_upper'] = upper
    data['BB_lower'] = lower
    data['BB_width'] = width
    data['BB_pos'] = pos

    # ========== ATR ==========
    data['ATR'] = _atr(data['high'], data['low'], data['close'], 14)
    data['ATR_pct'] = data['ATR'] / data['close'] * 100

    # ========== 成交量 ==========
    data['vol_ma5'] = data['volume'].rolling(5).mean()
    data['vol_ratio'] = data['volume'] / data['vol_ma5'].replace(0, np.nan)
    data['amt_ma5'] = data['amount'].rolling(5).mean()
    data['amt_ratio'] = data['amount'] / data['amt_ma5'].replace(0, np.nan)

    # ========== 动量 ==========
    for p in [3, 5, 10]:
        data[f'mom_{p}d'] = data['close'].pct_change(p) * 100

    # ========== 波动率 ==========
    data['volatility'] = data['ret_1d'].rolling(10).std() * 100

    # ========== 目标变量 ==========
    data['target'] = data['close'].pct_change(1).shift(-1) * 100

    return data


def create_sequences(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    将特征数据转换为时间序列窗口。

    Returns
    -------
    X_seq : shape [n_samples, window_size, n_features]
    y_seq : shape [n_samples]
    feature_columns : list[str]
    """
    exclude_cols = {'date', 'target', 'open', 'high', 'low',
                    'volume', 'amount', 'preclose', 'tradestatus',
                    'isST', 'pctChg', 'code', 'adjustflag'}
    feature_columns = [c for c in df.columns
                       if c not in exclude_cols and not c.startswith('target')
                       and df[c].dtype in ('float64', 'int64', 'float32')]

    valid = df[feature_columns + ['target']].dropna()
    if len(valid) < window_size + 1:
        return np.array([]), np.array([]), feature_columns

    values = valid[feature_columns].values
    targets = valid['target'].values

    X_seq, y_seq = [], []
    for i in range(len(values) - window_size):
        X_seq.append(values[i:i + window_size])
        y_seq.append(targets[i + window_size])

    return np.array(X_seq), np.array(y_seq), feature_columns


# =====================================================================
def _rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close, fast, slow, signal):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    dif = ef - es
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def _bollinger(close, period, std):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    upper = mid + s * std
    lower = mid - s * std
    width = (upper - lower) / mid * 100
    pos = (close - lower) / (upper - lower + 1e-10)
    return mid, upper, lower, width, pos


def _atr(high, low, close, period):
    prev = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev).abs(), (low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
