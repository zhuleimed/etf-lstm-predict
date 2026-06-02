"""
特征工程模块 — ETF LSTM 预测（精简版）

注意: baostock ETF 数据约 96 行（2026-01 起），所有滚动窗口限制在 20 天内。
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from config.etf_config import WINDOW_SIZE


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    对单只 ETF 计算精简特征集。

    Parameters
    ----------
    df : pd.DataFrame
        必须包含列: date, open, high, low, close, volume, amount

    Returns
    -------
    pd.DataFrame
        原数据 + 特征列 + target（次日收益率 %）
    """
    data = df.copy().sort_values('date').reset_index(drop=True)

    # ===== 1. 基础收益率 =====
    data['ret_1d'] = data['close'].pct_change() * 100
    data['ret_5d'] = data['close'].pct_change(5) * 100
    data['log_ret'] = np.log(data['close'] / data['close'].shift(1)) * 100
    data['amplitude'] = (data['high'] - data['low']) / data['close'].shift(1) * 100

    # ===== 2. 移动平均线（MA, 最长 20 天）=====
    for p in [5, 10, 20]:
        ma = data['close'].rolling(p).mean()
        data[f'MA{p}'] = ma
        data[f'MA{p}_dev'] = (data['close'] - ma) / ma * 100  # 偏离度

    # ===== 3. RSI(6) 和 RSI(12) =====
    data['RSI6'] = _rsi(data['close'], 6)
    data['RSI12'] = _rsi(data['close'], 12)

    # ===== 4. MACD(12,26,9) =====
    dif, dea, hist = _macd(data['close'], 12, 26, 9)
    data['MACD_DIF'] = dif
    data['MACD_DEA'] = dea
    data['MACD_HIST'] = hist

    # ===== 5. 布林带(20,2) =====
    mid, upper, lower, width, pos = _bollinger(data['close'], 20, 2)
    data['BB_mid'] = mid
    data['BB_upper'] = upper
    data['BB_lower'] = lower
    data['BB_width'] = width
    data['BB_pos'] = pos

    # ===== 6. ATR(14) =====
    data['ATR'] = _atr(data['high'], data['low'], data['close'], 14)
    data['ATR_pct'] = data['ATR'] / data['close'] * 100

    # ===== 7. 成交量特征 =====
    data['vol_ma5'] = data['volume'].rolling(5).mean()
    data['vol_ratio'] = data['volume'] / data['vol_ma5'].replace(0, np.nan)
    data['amt_ma5'] = data['amount'].rolling(5).mean()
    data['amt_ratio'] = data['amount'] / data['amt_ma5'].replace(0, np.nan)

    # ===== 8. 动量 (最长 10 天) =====
    for p in [3, 5, 10]:
        data[f'mom_{p}d'] = data['close'].pct_change(p) * 100

    # ===== 9. 波动率 =====
    data['volatility'] = data['ret_1d'].rolling(10).std() * 100

    # ===== 目标变量 =====
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
    X_seq : np.ndarray, shape [n_samples, window_size, n_features]
    y_seq : np.ndarray, shape [n_samples]
    feature_columns : list of str
    """
    exclude_cols = {'date', 'target', 'open', 'high', 'low',
                    'volume', 'amount', 'preclose', 'tradestatus',
                    'isST', 'pctChg', 'code', 'adjustflag'}
    feature_columns = [c for c in df.columns
                       if c not in exclude_cols and not c.startswith('target')
                       and df[c].dtype in ('float64', 'int64', 'float32')]

    # 去掉含 NaN 的行（特征计算需要前 window_size 行）
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
# 内部指标计算函数
# =====================================================================

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif, dea, hist


def _bollinger(close: pd.Series, period: int, std: int):
    mid = close.rolling(period).mean()
    s = close.rolling(period).std()
    upper = mid + s * std
    lower = mid - s * std
    width = (upper - lower) / mid * 100
    pos = (close - lower) / (upper - lower + 1e-10)
    return mid, upper, lower, width, pos


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
