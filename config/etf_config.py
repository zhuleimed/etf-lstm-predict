"""
ETF LSTM-Transformer 预测模拟盘系统 — 全局配置

所有路径、参数、常量集中管理。
"""

import os
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.dirname(PROJECT_DIR)
QUANT_DIR = os.path.dirname(CODE_DIR)

# ============================================================================
# ETF 配置
# ============================================================================
ETF_CODES = ['510300', '510310', '510330', '159919']
ETF_DESCRIPTIONS = {
    '510300': '华泰柏瑞沪深300ETF',
    '510310': '易方达沪深300ETF',
    '510330': '华夏沪深300ETF',
    '159919': '嘉实沪深300ETF',
}

INITIAL_CAPITAL_PER_ETF = 10_000.0
INITIAL_CAPITAL = INITIAL_CAPITAL_PER_ETF * len(ETF_CODES)

# ============================================================================
# 交易成本参数
# ============================================================================
SLIPPAGE = 0.003
COMMISSION_RATE = 0.0005
POSITION_PCT = 0.95
ETF_TAX_FREE = True

# 涨跌信号阈值（0.8%，覆盖滑点+佣金+安全边际）
SIGNAL_THRESHOLD = 0.008

MIN_TRADE_UNIT = 100

# ============================================================================
# 数据与路径
# ============================================================================
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
STATE_FILE = os.path.join(OUTPUT_DIR, 'state.json')

# ============================================================================
# 模型与训练参数
# ============================================================================
# 数据源：baostock ETF 约 96 行（2026-01 起），窗口 20 天可得 ~50 样本
WINDOW_SIZE = 20
PREDICTION_HORIZON = 1
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# ---- 完整 LSTM-Transformer 模型参数（小数据优化）----
PRODUCTION_MODEL_PARAMS = {
    'lstm_hidden': 48,
    'lstm_layers': 1,
    'transformer_dim': 48,
    'nhead': 4,
    'num_transformer_layers': 1,
    'fc_hidden': 24,
    'additional_fc_layers': 0,
    'dropout': 0.3,
    'lr': 0.0005,
    'batch_size': 8,
    'epochs': 100,
}

# ---- 简化版 LSTM 模型参数（小数据推荐）----
SIMPLE_MODEL_PARAMS = {
    'lstm_hidden': 32,
    'lstm_layers': 1,
    'transformer_dim': 32,
    'nhead': 4,
    'num_transformer_layers': 1,
    'fc_hidden': 16,
    'additional_fc_layers': 0,
    'dropout': 0.2,
    'lr': 0.0005,
    'batch_size': 8,
    'epochs': 50,
}

# 小数据量下推荐简化版（过拟合风险更低）
# 测试=True，生产建议保持 True
USE_SIMPLE_MODEL = True

# 模型重训练间隔（每 5 个交易日）
MODEL_RETRAIN_INTERVAL = 5
MODEL_MIN_TRAIN_DAYS = 30

# ============================================================================
# 并行计算
# ============================================================================
# 训练时最多同时训练 4 只 ETF（线程池）
MAX_WORKERS = 4

# ============================================================================
# 设备
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# ============================================================================
# 技术指标参数
# ============================================================================
MA_PERIODS = [5, 10, 20]
RSI_PERIODS = [6, 12]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLL_PERIOD = 20
BOLL_STD = 2

# ============================================================================
# WxPusher 推送配置
# ============================================================================
WXPUSHER_TOKEN = 'AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX'
WXPUSHER_UIDS = ['<uids>']
WXPUSHER_TOPIC_IDS = ['39277']

# ============================================================================
# 风控
# ============================================================================
STOP_LOSS_PCT = 0.05

# ============================================================================
# 数据获取超时
# ============================================================================
FETCH_TIMEOUT = 60


def ensure_dirs():
    for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
