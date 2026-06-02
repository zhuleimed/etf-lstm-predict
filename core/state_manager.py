"""
状态管理模块 — ETF LSTM 预测模拟盘

JSON 状态持久化（原子写入）+ 状态机流转
"""

import json
import os
import tempfile
from datetime import date
from typing import Any, Dict, List, Optional

from config.etf_config import (
    INITIAL_CAPITAL, ETF_CODES, STATE_FILE,
)


class StateManager:
    """ETF 模拟盘状态管理器"""

    def __init__(self, state_file_path: str = STATE_FILE):
        self._path = state_file_path
        self._data: Dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def load(self) -> Dict[str, Any]:
        if os.path.exists(self._path):
            try:
                with open(self._path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                default = self._default_state()
                for key, val in default.items():
                    if key not in self._data:
                        self._data[key] = val
            except (json.JSONDecodeError, IOError):
                self._data = self._default_state()
        else:
            self._data = self._default_state()
        return self._data

    def save(self):
        self._data['last_update_date'] = date.today().isoformat()
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix='.json', prefix='state_',
            dir=os.path.dirname(self._path),
        )
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2,
                          default=str)
            os.replace(tmp_path, self._path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @property
    def current_phase(self) -> str:
        return self._data.get('current_phase', 'idle')

    @current_phase.setter
    def current_phase(self, phase: str):
        valid = {'idle', 'running'}
        if phase not in valid:
            raise ValueError(f'无效 phase: {phase}')
        self._data['current_phase'] = phase

    @property
    def portfolio(self) -> Dict[str, Any]:
        return self._data.get('portfolio', {})

    @property
    def model_info(self) -> Dict[str, str]:
        """{ETF_CODE: last_trained_date}"""
        return self._data.get('model_info', {})

    # ------------------------------------------------------------------
    # 模型信息
    # ------------------------------------------------------------------

    def set_model_trained(self, etf_code: str):
        models = self._data.setdefault('model_info', {})
        models[etf_code] = date.today().isoformat()
        self.save()

    # ------------------------------------------------------------------
    # 模拟盘持仓更新
    # ------------------------------------------------------------------

    def update_portfolio(self, cash: float, positions: Dict,
                         pending_orders: Dict[str, str],
                         portfolio_value: float):
        pf = self._data.setdefault('portfolio', {})
        pf['cash'] = round(cash, 2)
        pf['positions'] = positions
        pf['pending_orders'] = pending_orders
        pf['initial_capital'] = pf.get('initial_capital', INITIAL_CAPITAL)
        pf['_last_portfolio_value'] = portfolio_value

    def add_trade(self, entry: Dict, max_days: int = 180):
        """
        追加交易记录，自动清理超过 max_days 天的旧记录。
        """
        log = self._data.setdefault('trade_log', [])
        log.append(entry)

        # 按日期清理：仅保留最近 max_days 天
        if len(log) > 100:  # 延迟到有足够记录后再清理
            cutoff = (date.today().isoformat())
            from datetime import timedelta
            cutoff_date = date.today() - timedelta(days=max_days)
            cutoff_str = cutoff_date.isoformat()
            self._data['trade_log'] = [
                t for t in log
                if t.get('date', '').startswith('20') and t.get('date', '') >= cutoff_str
            ]

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def needs_retrain(self, etf_code: str, interval_days: int = 5) -> bool:
        """检查某 ETF 是否需要重训练。"""
        from datetime import datetime
        last = self.model_info.get(etf_code)
        if last is None:
            return True
        try:
            last_dt = datetime.strptime(str(last)[:10], '%Y-%m-%d').date()
            return (date.today() - last_dt).days >= interval_days
        except (ValueError, TypeError):
            return True

    def is_first_run(self) -> bool:
        return not self.model_info

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {
            'version': 1,
            'current_phase': 'idle',
            'model_info': {},
            'portfolio': {
                'cash': INITIAL_CAPITAL,
                'initial_capital': INITIAL_CAPITAL,
                'positions': {},
                'pending_orders': {},
            },
            'benchmark': {},  # {code: {shares, initial_price}}
            'trade_log': [],
            'last_update_date': None,
        }
