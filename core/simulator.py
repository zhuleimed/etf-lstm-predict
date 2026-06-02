"""
模拟盘引擎 — ETF LSTM 预测

每日执行：
  1. 交易日判断 → ETF 数据加载 → 模型预测 → 信号生成
  2. 执行昨日待处理订单
  3. 生成明日信号
  4. 计算策略 vs 基准收益对比
  5. 返回摘要供推送
"""

from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.etf_config import (
    ETF_CODES, ETF_DESCRIPTIONS, ETF_TAX_FREE,
    INITIAL_CAPITAL, INITIAL_CAPITAL_PER_ETF,
    MIN_TRADE_UNIT, POSITION_PCT,
    SLIPPAGE, COMMISSION_RATE, TAX_RATE,
    SIGNAL_THRESHOLD, STOP_LOSS_PCT,
    WINDOW_SIZE,
)
from core.hs300_utils import is_trading_day, fetch_etf_from_baostock
from core.log_utils import get_logger
from model.lstm_transformer_predictor import LSTMTransformerPredictor

logger = get_logger(__name__)


class Simulator:
    """每日模拟盘引擎。"""

    def __init__(self):
        self.predictors: Dict[str, LSTMTransformerPredictor] = {}

    def run_daily(
        self,
        state_manager: Any,
        dry_run: bool = False,
    ) -> Optional[Dict]:
        """
        执行每日模拟盘流程。

        Parameters
        ----------
        state_manager : StateManager
            状态管理器
        dry_run : bool
            True = 只输出操作不修改状态

        Returns
        -------
        dict or None
            日报摘要，非交易日返回 None
        """
        # ---- 1. 交易日检查 ----
        if not is_trading_day():
            logger.info(f'[{date.today().isoformat()}] 非交易日，跳过模拟盘')
            return None

        today = date.today()
        today_str = today.isoformat()
        logger.info(f'[{today_str}] 开始 ETF 模拟盘…')

        # ---- 2. 加载所有 ETF 数据 ----
        etf_data = self._load_all_etf_data()
        if not etf_data:
            logger.warning(f'[{today_str}] 无法加载 ETF 数据')
            return None

        # ---- 3. 模型预测 — 每只 ETF 独立预测 ----
        predictions = self._predict_all(etf_data)

        # ---- 4. 从 state 获取持仓 ----
        pf = state_manager.portfolio
        cash = pf.get('cash', INITIAL_CAPITAL)
        positions = pf.get('positions', {})
        pending_orders = pf.get('pending_orders', {})
        initial_capital = pf.get('initial_capital', INITIAL_CAPITAL)

        # ---- 5. 执行昨日待处理订单 ----
        trades_today = []
        if pending_orders:
            for etf, action in list(pending_orders.items()):
                if etf not in etf_data:
                    continue
                row = etf_data[etf]['latest']

                if action == 'buy' and cash > 0:
                    trade = self._execute_buy(etf, row['open'], cash, positions)
                    if trade:
                        cash = trade['cash_after']
                        trades_today.append(trade)

                elif action == 'sell' and etf in positions:
                    trade = self._execute_sell(etf, row['open'], positions)
                    if trade:
                        cash += trade['net_revenue']
                        trades_today.append(trade)

            pending_orders = {}

        # ---- 6. 风控检查 ----
        for etf, pos in list(positions.items()):
            if etf not in etf_data:
                continue
            row = etf_data[etf]['latest']
            if self._should_force_sell(row, pos):
                trade = self._execute_sell(etf, row['open'], positions)
                if trade:
                    cash += trade['net_revenue']
                    trade['reason'] = 'risk_stop_loss'
                    trades_today.append(trade)

        # ---- 7. 根据预测生成明日信号 ----
        new_pending = {}
        for etf in ETF_CODES:
            if etf not in predictions or predictions[etf] is None:
                new_pending[etf] = 'hold'
                continue

            pred = predictions[etf]  # 预测涨跌幅（%）
            if pred > SIGNAL_THRESHOLD * 100 and etf not in positions:
                # 预测涨幅超过阈值且未持仓 → 买入信号
                new_pending[etf] = 'buy'
            elif pred < -SIGNAL_THRESHOLD * 100 and etf in positions:
                # 预测跌幅超过阈值且已持仓 → 卖出信号
                new_pending[etf] = 'sell'
            else:
                new_pending[etf] = 'hold'

        # ---- 8. 计算组合市值 ----
        portfolio_value = cash
        position_details = []
        for etf, pos in positions.items():
            if etf in etf_data:
                last_close = etf_data[etf]['latest']['close']
                market_value = pos['shares'] * last_close
                portfolio_value += market_value
                pnl_pct = (last_close - pos['avg_cost']) / pos['avg_cost']
                position_details.append({
                    'etf': etf,
                    'shares': pos['shares'],
                    'avg_cost': round(pos['avg_cost'], 4),
                    'last_close': round(last_close, 4),
                    'market_value': round(market_value, 2),
                    'pnl_pct': round(pnl_pct, 4),
                })

        cumulative_return = (portfolio_value - initial_capital) / initial_capital

        # ---- 9. 计算基准收益（等权买入持有所有 ETF）----
        benchmark_return = self._calc_benchmark_return(etf_data)
        excess_return = cumulative_return - benchmark_return

        # ---- 10. 更新状态 ----
        if not dry_run:
            state_manager.update_portfolio(cash, positions, new_pending, portfolio_value)
            for trade in trades_today:
                state_manager.add_trade(trade)
            state_manager.save()

        # ---- 11. 构建摘要 ----
        summary = {
            'date': today_str,
            'trades_today': trades_today,
            'positions': position_details,
            'cash': round(cash, 2),
            'portfolio_value': round(portfolio_value, 2),
            'initial_capital': initial_capital,
            'cumulative_return': round(cumulative_return, 4),
            'benchmark_return': round(benchmark_return, 4),
            'excess_return': round(excess_return, 4),
            'pending_orders': new_pending,
            'predictions': predictions,
            'model_status': self._get_model_status(),
            'dry_run': dry_run,
        }

        self._print_summary(summary)
        return summary

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _load_all_etf_data(self) -> Dict[str, Dict]:
        """加载所有 ETF 的最新数据。"""
        today = date.today()
        start = today.replace(year=today.year - 3)
        start_str = start.strftime('%Y-%m-%d')
        end_str = today.strftime('%Y-%m-%d')

        result = {}
        for code in ETF_CODES:
            try:
                df = fetch_etf_from_baostock(code, start_str, end_str)
                if df is not None and len(df) > 60:
                    result[code] = {
                        'df': df,
                        'latest': df.iloc[-1],
                    }
                else:
                    logger.warning(f'{code}: baostock 数据不足')
            except Exception as e:
                logger.warning(f'{code}: 数据获取失败: {e}')
        return result

    def _predict_all(self, etf_data: Dict) -> Dict[str, Optional[float]]:
        """
        对每只 ETF 做预测。

        Returns
        -------
        dict : {code: predicted_return_pct or None}
        """
        predictions = {}
        for code in ETF_CODES:
            if code not in etf_data:
                predictions[code] = None
                continue

            df = etf_data[code]['df']
            predictor = LSTMTransformerPredictor()

            # 加载已有模型（训练由 run_daily.py 的 train_all_etf_models 负责）
            if predictor.model_exists(code):
                loaded = predictor.load(code)
                if loaded:
                    pred = predictor.predict_next_day(df)
                    predictions[code] = pred
                    logger.info(f'  {code} 预测: {pred:.4f}%')
                    self.predictors[code] = predictor
                    continue

            # 无模型 → 无法预测
            predictions[code] = None
            logger.warning(f'  {code}: 无历史模型，请先执行 --train')
        return predictions

    def _execute_buy(self, etf: str, open_price: float,
                     cash: float, positions: Dict) -> Optional[Dict]:
        """执行买入。"""
        available = cash * POSITION_PCT
        exec_price = open_price * (1 + SLIPPAGE)

        raw_shares = int(available / exec_price)
        shares = (raw_shares // MIN_TRADE_UNIT) * MIN_TRADE_UNIT
        if shares == 0:
            return None

        gross_cost = shares * exec_price
        commission = max(gross_cost * COMMISSION_RATE, 5.0)
        total_cost = gross_cost + commission

        if total_cost > cash:
            return None

        if etf in positions:
            old = positions[etf]
            new_shares = old['shares'] + shares
            new_total = old['total_cost'] + gross_cost
            positions[etf] = {
                'shares': new_shares,
                'avg_cost': round(new_total / new_shares, 4),
                'total_cost': round(new_total, 2),
            }
        else:
            positions[etf] = {
                'shares': shares,
                'avg_cost': round(exec_price, 4),
                'total_cost': round(gross_cost, 2),
            }

        return {
            'date': date.today().isoformat(),
            'etf': etf,
            'action': 'buy',
            'price': round(exec_price, 4),
            'shares': shares,
            'cost': round(total_cost, 2),
            'commission': round(commission, 2),
            'cash_after': round(cash - total_cost, 2),
            'reason': 'signal_buy',
        }

    def _execute_sell(self, etf: str, open_price: float,
                      positions: Dict) -> Optional[Dict]:
        """执行卖出。"""
        if etf not in positions:
            return None

        pos = positions[etf]
        exec_price = open_price * (1 - SLIPPAGE)
        gross_revenue = pos['shares'] * exec_price
        commission = max(gross_revenue * COMMISSION_RATE, 5.0)

        # ETF 免印花税
        tax = 0 if ETF_TAX_FREE else gross_revenue * TAX_RATE

        net_revenue = gross_revenue - commission - tax
        pnl = round(net_revenue - pos['total_cost'], 2)

        del positions[etf]

        return {
            'date': date.today().isoformat(),
            'etf': etf,
            'action': 'sell',
            'price': round(exec_price, 4),
            'shares': pos['shares'],
            'net_revenue': round(net_revenue, 2),
            'commission': round(commission, 2),
            'tax': round(tax, 2),
            'pnl': pnl,
            'reason': 'signal_sell',
        }

    def _should_force_sell(self, row: pd.Series, position: Dict) -> bool:
        """止损检查。"""
        open_px = row.get('open', 0)
        avg_cost = position.get('avg_cost', 0)
        if open_px <= 0 or avg_cost <= 0:
            return False
        return open_px < avg_cost * (1 - STOP_LOSS_PCT)

    def _calc_benchmark_return(self, etf_data: Dict) -> float:
        """
        计算基准收益率。

        基准 = 等权买入持有所有 ETF 的累计收益。
        以当前最新收盘价计算。
        """
        total = 0.0
        count = 0
        for code in ETF_CODES:
            if code in etf_data:
                df = etf_data[code]['df']
                if len(df) >= 2:
                    first_close = df.iloc[0]['close']
                    last_close = df.iloc[-1]['close']
                    ret = (last_close - first_close) / first_close
                    total += ret
                    count += 1
        return total / count if count > 0 else 0.0

    def _get_model_status(self) -> Dict[str, str]:
        status = {}
        for code in ETF_CODES:
            from model.lstm_transformer_predictor import LSTMTransformerPredictor
            p = LSTMTransformerPredictor()
            if p.model_exists(code):
                status[code] = 'loaded'
            else:
                status[code] = 'none'
        return status

    def _print_summary(self, summary: Dict):
        """控制台输出摘要。"""
        print(f'\n{"=" * 55}')
        print(f'  📊 ETF LSTM 模拟盘日报 — {summary["date"]}')
        if summary.get('dry_run'):
            print(f'  ⚠ DRY RUN 模式')
        print(f'{"=" * 55}')

        if summary['trades_today']:
            print(f'\n  当日操作:')
            for t in summary['trades_today']:
                action = '🟢 买入' if t['action'] == 'buy' else '🔴 卖出'
                print(f'    {action} {t["etf"]}: {t["shares"]}份 @ {t["price"]:.4f}  '
                      f'成本={t.get("cost", 0):.2f}  '
                      f'盈亏={t.get("pnl", 0):+.2f}')
        else:
            print(f'\n  当日无操作')

        if summary['positions']:
            print(f'\n  持仓摘要:')
            for p in summary['positions']:
                sign = '+' if p['pnl_pct'] >= 0 else ''
                print(f'    {p["etf"]}: {p["shares"]}份  '
                      f'成本={p["avg_cost"]:.4f}  '
                      f'现价={p["last_close"]:.4f}  '
                      f'({sign}{p["pnl_pct"]:.2%})')
        else:
            print(f'\n  空仓')

        print(f'\n  账户摘要:')
        print(f'    总资产:   {summary["portfolio_value"]:,.2f}')
        print(f'    现金:     {summary["cash"]:,.2f}')
        print(f'    策略收益: {summary["cumulative_return"]:+.2%}')
        print(f'    基准收益: {summary["benchmark_return"]:+.2%}')
        print(f'    超额收益: {summary["excess_return"]:+.2%}')

        if summary.get('predictions'):
            print(f'\n  模型预测(次日涨跌幅):')
            for code, pred in summary['predictions'].items():
                if pred is not None:
                    arrow = '🟢' if pred > 0 else '🔴'
                    print(f'    {arrow} {code}: {pred:+.4f}%')
                else:
                    print(f'    ⚪ {code}: 无预测')

        print(f'{"=" * 55}')
