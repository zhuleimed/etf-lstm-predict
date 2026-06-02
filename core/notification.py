"""
消息推送模块 — ETF LSTM 预测模拟盘

WxPusher 微信推送：日报、异常告警、训练完成通知
"""

from datetime import date
from typing import Any, Dict, List, Optional

from wxpusher import WxPusher

from config.etf_config import WXPUSHER_TOKEN, WXPUSHER_UIDS, WXPUSHER_TOPIC_IDS


def _send(message: str):
    try:
        WxPusher.send_message(
            message,
            uids=WXPUSHER_UIDS,
            topic_ids=WXPUSHER_TOPIC_IDS,
            token=WXPUSHER_TOKEN,
        )
    except Exception as e:
        print(f'[WxPusher] 推送失败: {e}')


def push_daily_report(summary: Dict[str, Any]):
    """
    推送每日模拟盘日报。

    Parameters
    ----------
    summary : dict
        Simulator.run_daily() 返回的日报摘要
    """
    today = summary.get('date', date.today().isoformat())

    lines = [
        f'📊 ETF LSTM 预测 · 模拟盘日报',
        f'日期: {today}',
    ]

    if summary.get('dry_run'):
        lines.append('⚠ DRY RUN 模式（未修改状态）')

    # ---- 当日操作 ----
    trades = summary.get('trades_today', [])
    lines.append('')
    lines.append('── 当日操作 ──')
    if trades:
        for t in trades:
            if t['action'] == 'buy':
                lines.append(
                    f'🟢 买入 {t["etf"]}: '
                    f'{t["shares"]}份 @ {t["price"]:.4f}  '
                    f'成本={t["cost"]:.2f}'
                )
            else:
                lines.append(
                    f'🔴 卖出 {t["etf"]}: '
                    f'{t["shares"]}份 @ {t["price"]:.4f}  '
                    f'盈亏={t.get("pnl", 0):+.2f}  '
                    f'({t.get("reason", "")})'
                )
    else:
        lines.append('  无操作')

    # ---- 持仓摘要 ----
    positions = summary.get('positions', [])
    lines.append('')
    lines.append('── 持仓摘要 ──')
    if positions:
        for p in positions:
            sign = '+' if p['pnl_pct'] >= 0 else ''
            lines.append(
                f'  {p["etf"]}: {p["shares"]}份  '
                f'成本={p["avg_cost"]:.4f}  '
                f'现价={p["last_close"]:.4f}  '
                f'({sign}{p["pnl_pct"]:.2%})'
            )
    else:
        lines.append('  空仓')

    # ---- 账户摘要 ----
    lines.append('')
    lines.append('── 账户摘要 ──')
    lines.append(f'总资产: {summary["portfolio_value"]:,.2f}')
    lines.append(f'现金:   {summary["cash"]:,.2f}')
    cum_ret = summary['cumulative_return']
    sign = '+' if cum_ret >= 0 else ''
    lines.append(f'策略累计收益: {sign}{cum_ret:.2%}')

    # ---- 基准对比 ----
    bench_ret = summary.get('benchmark_return', 0)
    bench_sign = '+' if bench_ret >= 0 else ''
    lines.append(f'基准累计收益: {bench_sign}{bench_ret:.2%}')

    excess = summary.get('excess_return', 0)
    excess_sign = '+' if excess >= 0 else ''
    lines.append(f'超额收益:     {excess_sign}{excess:.2%}')

    # ---- 明日信号 ----
    pending = summary.get('pending_orders', {})
    if pending:
        active = {s: a for s, a in pending.items() if a != 'hold'}
        if active:
            lines.append('')
            lines.append('── 明日信号 ──')
            for etf, action in active.items():
                emoji = '🟢' if action == 'buy' else '🔴'
                lines.append(f'  {emoji} {etf}: {action}')

    # ---- 模型状态 ----
    model_status = summary.get('model_status', {})
    if model_status:
        lines.append('')
        lines.append('── 模型状态 ──')
        for etf, status in model_status.items():
            if status == 'trained':
                lines.append(f'  ✓ {etf}: 已训练')
            elif status == 'loaded':
                lines.append(f'  ✓ {etf}: 加载已有模型')
            elif status == 'failed':
                lines.append(f'  ✗ {etf}: 训练失败')

    _send('\n'.join(lines))


def push_error(error_msg: str, phase: str = ''):
    """推送异常告警。"""
    lines = [
        '⚠ ETF LSTM 预测 · 异常告警',
        f'日期: {date.today().isoformat()}',
    ]
    if phase:
        lines.append(f'阶段: {phase}')
    lines.append('')
    lines.append(f'错误: {error_msg}')
    _send('\n'.join(lines))


def push_training_complete(results: Dict[str, str]):
    """推送模型训练完成通知。"""
    lines = [
        '🤖 ETF LSTM 模型训练完成',
        f'日期: {date.today().isoformat()}',
        '',
    ]
    for etf, status in results.items():
        emoji = '✓' if status == 'ok' else '✗'
        lines.append(f'  {emoji} {etf}: {status}')
    _send('\n'.join(lines))
