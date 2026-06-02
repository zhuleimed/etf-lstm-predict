#!/usr/bin/env python3
"""
run_daily.py — ETF LSTM 预测模拟盘系统主入口

用法:
  python run_daily.py                    # 自动判断（默认）
  python run_daily.py --phase 4          # 执行模拟盘
  python run_daily.py --train            # 强制重训练所有模型
  python run_daily.py --dry-run          # 模拟盘 dry-run

Cron 配置:
  # 每日模拟盘（交易日 21:00）
  0 21 * * 1-5 cd /path/to/016_etf_lstm_predict && \\
    /home/zhulei/anaconda3/envs/zhulei/bin/python run_daily.py \\
    >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
"""

import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime
from typing import Dict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config.etf_config import ensure_dirs, OUTPUT_DIR, ETF_CODES, MODEL_DIR
from core.state_manager import StateManager
from core.simulator import Simulator
from core.notification import push_daily_report, push_error, push_training_complete
from core.hs300_utils import is_trading_day
from core.log_utils import get_logger
from model.lstm_transformer_predictor import LSTMTransformerPredictor

logger = get_logger(__name__)

LOCK_FILE = os.path.join(OUTPUT_DIR, '.run_etf.lock')


def _check_stale_process():
    """检查并清理残留的 run_daily.py 进程。"""
    my_pid = os.getpid()

    def _is_same_script(pid: int) -> bool:
        try:
            cmdline_path = f'/proc/{pid}/cmdline'
            if not os.path.exists(cmdline_path):
                return False
            with open(cmdline_path, 'rb') as f:
                raw = f.read()
            parts = raw.decode('utf-8', errors='replace').split('\0')
            if len(parts) < 2:
                return False
            if 'python' not in parts[0].lower():
                return False
            return any('run_daily.py' in p for p in parts[1:])
        except (OSError, IOError):
            return False

    try:
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == my_pid:
                continue
            if _is_same_script(pid):
                print(f'[启动] 发现残留进程 PID={pid}，正在清理…')
                try:
                    os.kill(pid, 15)
                    import time
                    time.sleep(0.5)
                    os.kill(pid, 0)
                    os.kill(pid, 9)
                except OSError:
                    pass
    except PermissionError:
        pass

    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                old_pid_str = f.read().strip()
            if old_pid_str and old_pid_str.isdigit():
                old_pid = int(old_pid_str)
                if old_pid != my_pid and _is_same_script(old_pid):
                    try:
                        os.kill(old_pid, 15)
                        import time
                        time.sleep(0.5)
                        os.kill(old_pid, 0)
                        os.kill(old_pid, 9)
                    except OSError:
                        pass
        except (ValueError, OSError):
            pass

    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    with open(LOCK_FILE, 'w') as f:
        f.write(str(my_pid))


def _cleanup_lock():
    try:
        if os.path.exists(LOCK_FILE):
            with open(LOCK_FILE) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(LOCK_FILE)
    except Exception:
        pass


def train_all_etf_models(force: bool = False) -> Dict[str, str]:
    """
    训练/重训练所有 ETF 的模型。

    说明：baostock 不支持并行连接，训练过程串行。
    每只 ETF 约 2~3 分钟，4 只共约 8~12 分钟，
    每日 cron 21:00 执行，时间充裕。

    Returns
    -------
    dict : {code: 'ok' | 'skipped' | 'failed'}
    """
    from datetime import date
    from core.hs300_utils import fetch_etf_from_baostock

    results = {}
    state = StateManager()
    today = date.today()
    start = today.replace(year=today.year - 3)
    start_str = start.strftime('%Y-%m-%d')
    end_str = today.strftime('%Y-%m-%d')

    print(f'\n{"=" * 50}')
    print(f'  模型训练 — {today.isoformat()}')
    print(f'  ETF: {", ".join(ETF_CODES)}')
    print(f'{"=" * 50}')

    for code in ETF_CODES:
        if not force and not state.needs_retrain(code, interval_days=5):
            print(f'  ⏭ {code}: 距上次训练不足 5 天，跳过')
            results[code] = 'skipped'
            continue

        print(f'\n  📥 加载 {code} 数据…', end=' ', flush=True)
        df = fetch_etf_from_baostock(code, start_str, end_str)
        if df is None or len(df) < 50:
            print(f'数据不足 ({len(df) if df is not None else 0} 行)，跳过')
            results[code] = 'failed'
            continue
        print(f'{len(df)} 行 ✓')

        print(f'  🏋️  训练 {code} 模型…')
        predictor = LSTMTransformerPredictor()
        success = predictor.train(df, use_simple=True)
        if success:
            predictor.save(code)
            state.set_model_trained(code)
            results[code] = 'ok'
            print(f'  ✓ {code}: 训练完成')
        else:
            results[code] = 'failed'
            print(f'  ✗ {code}: 训练失败')

    return results


def run_daily_simulation(state, simulator, dry_run=False):
    """执行每日模拟盘。"""
    try:
        # 检查并训练缺失/过期的模型
        need_train = any(state.needs_retrain(code, interval_days=5)
                         for code in ETF_CODES)
        any_missing = any(
            not LSTMTransformerPredictor().model_exists(code)
            for code in ETF_CODES
        )
        if need_train or any_missing:
            if any_missing:
                logger.info('检测到新ETF无历史模型')
            logger.info('开始模型训练…')
            train_all_etf_models(force=False)

        summary = simulator.run_daily(state_manager=state, dry_run=dry_run)

        if summary is None:
            # 非交易日
            return

        if not dry_run:
            push_daily_report(summary)

    except Exception as e:
        logger.error(f'模拟盘异常: {e}', exc_info=True)
        push_error(str(e), '模拟盘')


def main():
    _check_stale_process()

    parser = argparse.ArgumentParser(
        description='016_etf_lstm_predict — ETF LSTM 预测模拟盘系统',
    )
    parser.add_argument('--phase', type=str, default='auto',
                        choices=['auto', '4'],
                        help='执行阶段。auto=自动判断（默认）')
    parser.add_argument('--dry-run', action='store_true',
                        help='dry-run（不修改状态）')
    parser.add_argument('--train', action='store_true',
                        help='强制重新训练所有模型')
    args = parser.parse_args()

    ensure_dirs()
    state = StateManager()
    simulator = Simulator()

    if args.train:
        results = train_all_etf_models(force=True)
        push_training_complete(results)
        logger.info('训练完成')
        _cleanup_lock()
        return

    # 自动判断
    if args.phase == 'auto':
        if is_trading_day():
            run_daily_simulation(state, simulator, dry_run=args.dry_run)
        else:
            logger.info(f'非交易日，跳过')
    elif args.phase == '4':
        run_daily_simulation(state, simulator, dry_run=args.dry_run)

    logger.info('执行完毕')
    _cleanup_lock()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        _cleanup_lock()
        raise
