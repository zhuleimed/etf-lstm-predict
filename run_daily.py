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


def _train_worker(code_df):
    """
    模块级 worker 函数：在独立线程中训练单只 ETF 模型。

    ThreadPoolExecutor 要求目标函数在模块级别定义（可 pickle）。
    """
    code, df, use_simple = code_df
    try:
        predictor = LSTMTransformerPredictor(use_simple=use_simple)
        ok = predictor.train(df)
        if ok:
            predictor.save(code)
            return code, 'ok', None
        return code, 'failed', '训练未收敛'
    except Exception as e:
        return code, 'failed', str(e)


def train_all_etf_models(force: bool = False) -> Dict[str, str]:
    """
    训练/重训练所有 ETF 的模型。

    流程：
      1. 串行加载数据（baostock，每只 ~2 秒，共 ~8 秒）
      2. 用 ThreadPoolExecutor 并行训练 4 只 ETF
      3. 串行更新 state（无竞态）

    总耗时：取决于训练最慢的那只 ETF（约 5~8 分钟/只）
    正式训练用完整 LSTM-Transformer 模型，测试时用简化版。

    Returns
    -------
    dict : {code: 'ok' | 'skipped' | 'failed'}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import date
    from config.etf_config import USE_SIMPLE_MODEL, MAX_WORKERS
    from core.hs300_utils import fetch_etf_data

    results = {}
    state = StateManager()
    today = date.today()

    print(f'\n{"=" * 50}')
    print(f'  模型训练 — {today.isoformat()}')
    model_tag = '简化版 LSTM' if USE_SIMPLE_MODEL else '完整版 LSTM-Transformer'
    print(f'  模型: {model_tag}')
    print(f'  ETF: {", ".join(ETF_CODES)}')
    print(f'{"=" * 50}')

    # Step 1: 确定需要训练的 ETF
    codes_to_train = []
    for code in ETF_CODES:
        if not force and not state.needs_retrain(code, interval_days=5):
            print(f'  ⏭ {code}: 距上次训练不足 5 天，跳过')
            results[code] = 'skipped'
        else:
            codes_to_train.append(code)

    if not codes_to_train:
        print('  所有模型无需训练 ✓')
        return results

    # Step 2: 串行加载数据（baostock 不支持并行连接）
    print(f'\n  📥 加载 {len(codes_to_train)} 只 ETF 数据（baostock）…')
    data_map = {}
    for code in codes_to_train:
        df = fetch_etf_data(code)
        if df is not None and len(df) > 50:
            data_map[code] = df
            print(f'    ✓ {code}: {len(df)} 行')
        else:
            print(f'    ✗ {code}: 数据不足 ({len(df) if df is not None else 0} 行)')
            results[code] = 'failed'

    if not data_map:
        return results

    # Step 3: 线程池并行训练
    print(f'\n  🏋️  并行训练 {len(data_map)} 个模型 ({model_tag})…')
    print(f'      (ThreadPoolExecutor, max_workers={MAX_WORKERS})')

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(data_map))) as ex:
        futures = {
            ex.submit(_train_worker, (code, df, USE_SIMPLE_MODEL)): code
            for code, df in data_map.items()
        }
        for f in as_completed(futures):
            code = futures[f]
            try:
                code2, status, err = f.result()
                if status == 'ok':
                    state.set_model_trained(code2)
                    results[code2] = 'ok'
                    print(f'    ✓ {code2}: 训练完成')
                else:
                    results[code2] = 'failed'
                    print(f'    ✗ {code2}: {err}')
            except Exception as e:
                print(f'    ✗ {code}: {e}')
                results[code] = 'failed'

    ok_count = sum(1 for v in results.values() if v == 'ok')
    print(f'\n  📊 {ok_count}/{len(results)} 训练成功')
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
