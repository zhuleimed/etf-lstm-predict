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
# 沪深300 ETF 代码列表（用 baostock 时自动加 sh./sz. 前缀）
ETF_CODES = ['510300', '510310', '510330', '159919']
ETF_DESCRIPTIONS = {
    '510300': '华泰柏瑞沪深300ETF',
    '510310': '易方达沪深300ETF',
    '510330': '华夏沪深300ETF',
    '159919': '嘉实沪深300ETF',
}

# 初始资金（每只 ETF 10,000 元）
INITIAL_CAPITAL_PER_ETF = 10_000.0
INITIAL_CAPITAL = INITIAL_CAPITAL_PER_ETF * len(ETF_CODES)

# ============================================================================
# 交易成本参数（与项目 A 保持一致）
# ============================================================================
SLIPPAGE = 0.003       # 滑点 0.3%
COMMISSION_RATE = 0.0005  # 佣金 万分之五（最低 5 元）
TAX_RATE = 0.001       # 印花税 千分之一（ETF 免印花税，此处保留用于计算框架兼容）
POSITION_PCT = 0.95    # 仓位比例 95%

# ETF 免印花税，但保留参数不影响计算
ETF_TAX_FREE = True     # ETF 交易免印花税

# 涨跌信号阈值 = 滑点 + 佣金 ≈ 0.36%
SIGNAL_THRESHOLD = 0.0036

# ETF 交易单位（100 份 = 1 手）
MIN_TRADE_UNIT = 100

# ============================================================================
# 数据与路径
# ============================================================================
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')
STATE_FILE = os.path.join(OUTPUT_DIR, 'state.json')

# ============================================================================
# 模型与训练参数（移植自项目 B，简化版）
# ============================================================================
WINDOW_SIZE = 30            # 历史窗口天数（baostock ETF 数据约 96 行，30 天窗口可获得 ~50 样本）
PREDICTION_HORIZON = 1      # 预测次日涨跌幅
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50          # 简化版模型 50 epoch（10~20 即早停，实际约 3 分钟/ETF）
EARLY_STOPPING_PATIENCE = 10

# 默认模型参数（不使用贝叶斯优化）
DEFAULT_MODEL_PARAMS = {
    'lstm_hidden': 32,      # 从 64 减半 → 更快
    'lstm_layers': 1,
    'transformer_dim': 32,  # 从 64 减半 → 更快
    'nhead': 4,
    'num_transformer_layers': 1,
    'fc_hidden': 16,        # 从 32 减半 → 更快
    'additional_fc_layers': 0,
    'dropout': 0.2,
    'lr': 0.0005,
    'batch_size': 8,        # 从 16 减半 → 每 epoch 更快
    'epochs': 30,            # 从 50 减少 → 更早收敛
}

# 模型重训练间隔（每 5 个交易日）
MODEL_RETRAIN_INTERVAL = 5
MODEL_MIN_TRAIN_DAYS = 30   # 最小训练数据天数

# 最大历史数据天数（约 3 年）
MAX_HISTORY_DAYS = 800
MIN_HISTORY_DAYS = 180

# ============================================================================
# 并行计算（32 核）
# ============================================================================
MAX_WORKERS = 32

# ============================================================================
# 设备
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# ============================================================================
# 技术指标参数
# ============================================================================
MA_PERIODS = [5, 10, 20, 30, 60]
RSI_PERIODS = [6, 12, 24]
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
STOP_LOSS_PCT = 0.05   # 止损 5%

# ============================================================================
# 调用 baostock 的超时（秒）
# ============================================================================
BAOSTOCK_TIMEOUT = 45


def ensure_dirs():
    """确保所有输出目录存在。"""
    for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
