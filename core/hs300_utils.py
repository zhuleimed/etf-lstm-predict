"""
数据获取模块 — ETF LSTM 预测

职责：
  1. 交易日判断（baostock API）
  2. ETF 日线数据获取（baostock, 含 45s 超时保护）

注意：akshare/efinance 等基于东方财富的库在该服务器被屏蔽，
      目前只能用 baostock 获取约 96 行历史数据。
"""

from datetime import date, datetime
from typing import Optional

import pandas as pd

from config.etf_config import FETCH_TIMEOUT, ETF_CODES


def is_trading_day() -> bool:
    """判断今天是否为交易日（baostock API）。"""
    import baostock as bs
    today_str = date.today().strftime('%Y-%m-%d')
    try:
        lg = bs.login()
        if lg.error_code != '0':
            print(f'[交易日判断] baostock 登录失败，回退到工作日判断')
            return date.today().weekday() < 5
        rs = bs.query_trade_dates(start_date=today_str, end_date=today_str)
        is_trade = False
        if rs.error_code == '0':
            while rs.next():
                row = rs.get_row_data()
                if len(row) >= 2 and row[0] == today_str and row[1] == '1':
                    is_trade = True
        bs.logout()
        return is_trade
    except Exception as e:
        print(f'[交易日判断] baostock 查询异常: {e}')
        return date.today().weekday() < 5


def fetch_etf_data(
    etf_code: str,
    timeout_seconds: int = FETCH_TIMEOUT,
) -> Optional[pd.DataFrame]:
    """
    从 baostock 获取 ETF 日线数据（含 45s 超时保护）。

    Parameters
    ----------
    etf_code : str
        ETF 代码
    timeout_seconds : int
        API 超时秒数

    Returns
    -------
    pd.DataFrame or None
    """
    import baostock as bs
    import signal

    if etf_code.startswith('159'):
        bs_code = f'sz.{etf_code}'
    else:
        bs_code = f'sh.{etf_code}'

    class _Timeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise _Timeout(f'baostock API 超时 ({timeout_seconds}s)')

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_seconds)

    try:
        lg = bs.login()
        if lg.error_code != '0':
            print(f'[baostock] 登录失败: {lg.error_msg}')
            return None

        # 反查: baostock 只回溯约 5 个月（96 行）
        fields = ('date,code,open,high,low,close,preclose,'
                  'volume,amount,adjustflag,turn,tradestatus,pctChg,isST')
        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date='2020-01-01',
            end_date=date.today().strftime('%Y-%m-%d'),
            frequency='d', adjustflag='1',
        )

        if rs.error_code != '0':
            print(f'[baostock] {bs_code} 查询失败: {rs.error_msg}')
            bs.logout()
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        bs.logout()

        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)
        for col in ['code', 'adjustflag']:
            if col in df.columns:
                df = df.drop(columns=[col])

        numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                        'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'tradestatus' in df.columns:
            df['tradestatus'] = pd.to_numeric(
                df['tradestatus'], errors='coerce').fillna(1).astype(int)
        else:
            df['tradestatus'] = 1

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['tradestatus'] == 1].copy()
        df = df.dropna(subset=['date', 'open', 'close'])

        if len(df) == 0:
            return None

        # 统一输出列
        keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        return df[keep].reset_index(drop=True)

    except _Timeout:
        print(f'[baostock] {bs_code}: 超时 ({timeout_seconds}s)')
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_all_etf_data(lookback_years: int = 5) -> dict:
    """
    加载所有 ETF 的历史数据。

    Returns
    -------
    dict : {code: DataFrame}
    """
    result = {}
    print(f'\n[数据加载] 获取 {len(ETF_CODES)} 只 ETF 数据…')
    for code in ETF_CODES:
        df = fetch_etf_data(code)
        if df is not None and len(df) > 50:
            result[code] = df
            print(f'  ✓ {code}: {len(df)} 行')
        else:
            print(f'  ✗ {code}: 数据不足')
    print(f'  完成: {len(result)}/{len(ETF_CODES)}')
    return result
