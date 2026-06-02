"""
沪深300 ETF 工具模块 — baostock API 封装

职责：
  1. 交易日判断（baostock API）
  2. ETF 日线数据获取（baostock API，含超时保护）
  3. 指数数据加载
"""

from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd

from config.etf_config import BAOSTOCK_TIMEOUT, ETF_CODES


def is_trading_day() -> bool:
    """
    判断今天是否为交易日。
    通过 baostock API 的 query_trade_dates 查询。
    """
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
        print(f'[交易日判断] baostock 查询异常: {e}，回退到工作日判断')
        return date.today().weekday() < 5


def fetch_etf_from_baostock(
    etf_code: str,
    start_date: str,
    end_date: str,
    timeout_seconds: int = BAOSTOCK_TIMEOUT,
) -> Optional[pd.DataFrame]:
    """
    从 baostock API 获取 ETF 的日线数据，含超时保护。

    Parameters
    ----------
    etf_code : str
        ETF 代码（如 510300）
    start_date : str
        开始日期 'YYYY-MM-DD'
    end_date : str
        结束日期 'YYYY-MM-DD'
    timeout_seconds : int
        单次 API 调用超时秒数，防止网络挂起

    Returns
    -------
    pd.DataFrame or None
        列: date, open, high, low, close, volume, amount, pctChg
    """
    import baostock as bs
    import signal

    # baostock 中 ETF 代码统一加 sz. 前缀（深市ETF）或 sh.（沪市ETF）
    # 510xxx→上海, 159xxx→深圳
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

        fields = ('date,code,open,high,low,close,preclose,'
                  'volume,amount,adjustflag,turn,tradestatus,pctChg,isST')
        rs = bs.query_history_k_data_plus(
            bs_code, fields,
            start_date=start_date, end_date=end_date,
            frequency='d', adjustflag='3',  # 前复权
        )

        if rs.error_code != '0':
            print(f'[baostock] 获取 {bs_code} 数据失败: {rs.error_msg}')
            bs.logout()
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        bs.logout()

        if not data_list:
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 清理
        for col in ['code', 'adjustflag']:
            if col in df.columns:
                df = df.drop(columns=[col])

        numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                        'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'tradestatus' in df.columns:
            df['tradestatus'] = pd.to_numeric(df['tradestatus'], errors='coerce').fillna(1).astype(int)
        else:
            df['tradestatus'] = 1

        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 过滤非交易日
        if 'tradestatus' in df.columns:
            df = df[df['tradestatus'] == 1]

        # 关键列
        required = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = df.dropna(subset=required)
        if len(df) == 0:
            return None

        return df.reset_index(drop=True)

    except _Timeout:
        print(f'[baostock] 获取 {bs_code} 数据超时 ({timeout_seconds}s)，跳过')
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def load_all_etf_data(lookback_years: int = 3) -> dict:
    """
    加载所有 ETF 的历史数据。

    Parameters
    ----------
    lookback_years : int
        回溯年数

    Returns
    -------
    dict : {code: DataFrame}
    """
    today = date.today()
    start = today.replace(year=today.year - lookback_years)
    start_str = start.strftime('%Y-%m-%d')
    end_str = today.strftime('%Y-%m-%d')

    result = {}
    for code in ETF_CODES:
        df = fetch_etf_from_baostock(code, start_str, end_str)
        if df is not None and len(df) > 60:
            result[code] = df
            print(f'  ✓ {code}: {len(df)} 行')
        else:
            print(f'  ✗ {code}: 数据不足')
    return result
