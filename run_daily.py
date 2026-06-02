#!/usr/bin/env python3
"""
run_daily.py — ETF LSTM 预测模拟盘系统主入口

用法:
  python run_daily.py                    # 自动判断（默认）
  python run_daily.py --phase 4          # 执行模拟盘
  python run_daily.py --train            # 强制重训练所有模型
  python run_daily.py --dry-run          # 模拟盘 dry-run

Cron 配置:
  # 每日模拟盘（交易日 21:10，与项目 A 21:00 错开 10 分钟）
  10 21 * * 1-5 cd /path/to/016_etf_lstm_predict && \\
    /home/zhulei/anaconda3/envs/zhulei/bin/python run_daily.py \\
    >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config.etf_config import (
    ensure_dirs, OUTPUT_DIR, ETF_CODES, TRADING_ETF,
    MODEL_DIR, STATE_FILE, print_config,
)
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
    训练单一合并模型（4 只 ETF 数据合并训练）。

    流程：
      1. 串行加载 4 只 ETF 数据
      2. 调用 train_combined() 合并特征 → 训练单一 LSTM 模型
      3. 保存为 combined 模型

    4× 数据量 → 模型更稳健，只交易 510300。
    """
    from datetime import date
    from config.etf_config import USE_SIMPLE_MODEL
    from core.hs300_utils import fetch_etf_data

    results = {}
    state = StateManager()
    today = date.today()

    print(f'\n{"=" * 50}')
    print(f'  模型训练 — {today.isoformat()}')
    model_tag = '简化版 LSTM' if USE_SIMPLE_MODEL else '完整版 LSTM-Transformer'
    print(f'  模型: {model_tag}（4 ETF 合并训练 → 单一模型）')
    print(f'  数据源: {", ".join(ETF_CODES)}')
    print(f'  交易:   {TRADING_ETF}')
    print(f'{"=" * 50}')

    # 检查是否需要训练
    if not force and not state.needs_retrain('combined', interval_days=5):
        print(f'  距上次训练不足 5 天，跳过')
        results['combined'] = 'skipped'
        return results

    # 加载 4 只 ETF 数据
    print(f'\n  📥 加载 {len(ETF_CODES)} 只 ETF 数据（baostock）…')
    valid_dfs = []
    for code in ETF_CODES:
        df = fetch_etf_data(code)
        if df is not None and len(df) > 50:
            valid_dfs.append(df)
            print(f'    ✓ {code}: {len(df)} 行')
        else:
            print(f'    ✗ {code}: 数据不足')

    if len(valid_dfs) < 2:
        print('  有效数据不足 2 只 ETF，训练取消')
        results['combined'] = 'failed'
        return results

    total_rows = sum(len(df) for df in valid_dfs)
    print(f'  合并数据: {len(valid_dfs)} 只 ETF, 共 {total_rows} 行')

    # 训练单一合并模型
    print(f'\n  🏋️  训练合并模型 ({model_tag})…')
    predictor = LSTMTransformerPredictor(use_simple=USE_SIMPLE_MODEL)
    success = predictor.train_combined(valid_dfs, use_simple=USE_SIMPLE_MODEL)

    if success:
        predictor.save('combined')
        state.set_model_trained('combined')
        results['combined'] = 'ok'
        print(f'  ✓ combined 模型训练完成')
    else:
        results['combined'] = 'failed'
        print(f'  ✗ combined 模型训练失败')

    return results


def run_daily_simulation(state, simulator, dry_run=False):
    """执行每日模拟盘。"""
    try:
        # 检查合并模型是否需要训练
        need_train = state.needs_retrain('combined', interval_days=5)
        any_missing = not LSTMTransformerPredictor().model_exists('combined')
        if need_train or any_missing:
            if any_missing:
                logger.info('检测到合并模型不存在，开始训练…')
            else:
                logger.info('合并模型已过期，开始重训练…')
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


def run_health_check():
    """健康检查：验证模型、数据、状态文件是否正常。"""
    from core.hs300_utils import fetch_etf_data

    print(f'\n{"=" * 50}')
    print(f'  🔍 系统健康检查 — {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'{"=" * 50}')

    # 1. 状态文件
    state = StateManager()
    print(f'\n  📁 状态文件: {"✓" if os.path.exists(STATE_FILE) else "✗"}')

    # 2. 合并模型
    print(f'\n  🤖 合并模型 (4 ETF → 单一模型):')
    pred = LSTMTransformerPredictor()
    if pred.model_exists('combined'):
        loaded = pred.load('combined')
        if loaded:
            print(f'    ✓ combined: 模型文件存在且可加载')
            model_type = 'simple' if pred.use_simple else 'full'
            print(f'      类型={model_type}, 特征维度={len(pred.feature_columns)}')
        else:
            print(f'    ✗ combined: 模型文件损坏')
    else:
        print(f'    ⚪ combined: 无模型文件（请执行 --train）')

    # 3. 数据可用性
    print(f'\n  📡 ETF 数据:')
    for code in ETF_CODES:
        df = fetch_etf_data(code)
        if df is not None and len(df) > 50:
            print(f'    ✓ {code}: {len(df)} 行')
        else:
            print(f'    ✗ {code}: 数据不可用')

    # 4. 投资组合
    pf = state.portfolio
    print(f'\n  💰 投资组合:')
    print(f'    现金: {pf.get("cash", "N/A"):,.2f}')
    print(f'    持仓: {len(pf.get("positions", {}))} 只')
    print(f'    待执行订单: {pf.get("pending_orders", {})}')

    print(f'\n{"=" * 50}')
    print(f'  检查完成')
    print(f'{"=" * 50}')


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
    parser.add_argument('--check', action='store_true',
                        help='系统健康检查')
    parser.add_argument('--config', action='store_true',
                        help='显示当前运行参数')
    args = parser.parse_args()

    ensure_dirs()
    state = StateManager()
    simulator = Simulator()

    if args.config:
        print_config()
        _cleanup_lock()
        return

    if args.check:
        run_health_check()
        _cleanup_lock()
        return

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
