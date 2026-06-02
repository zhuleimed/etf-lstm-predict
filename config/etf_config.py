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
# 4 只沪深300 ETF 数据均用于训练（合并后 384 行）
# 交易时仅交易 TRADING_ETF（流动性最好的 510300）
ETF_CODES = ['510300', '510310', '510330', '159919']
TRADING_ETF = '510300'  # 实际交易的ETF
ETF_DESCRIPTIONS = {
    '510300': '华泰柏瑞沪深300ETF',
    '510310': '易方达沪深300ETF',
    '510330': '华夏沪深300ETF',
    '159919': '嘉实沪深300ETF',
}

# 初始资金（单一 ETF，10000 元）
INITIAL_CAPITAL = 10_000.0

# ============================================================================
# 交易成本参数
# ============================================================================
SLIPPAGE = 0.003
COMMISSION_RATE = 0.0005
POSITION_PCT = 0.95
ETF_TAX_FREE = True
SIGNAL_THRESHOLD = 0.008     # 0.8%
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
# 4 只 ETF 数据合并训练单一模型（96 × 4 = 384 行，得 ~200 样本）
WINDOW_SIZE = 20
PREDICTION_HORIZON = 1
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# ---- 完整 LSTM-Transformer 模型参数 ----
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

# ---- 简化版 LSTM 模型参数 ----
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

USE_SIMPLE_MODEL = True
MODEL_RETRAIN_INTERVAL = 5
MODEL_MIN_TRAIN_DAYS = 30

# ============================================================================
# 并行计算
# ============================================================================
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
FETCH_TIMEOUT = 60


def ensure_dirs():
    for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def print_config():
    """打印当前运行参数。"""
    print(f'\n{"=" * 50}')
    print(f'  ⚙️  系统运行参数')
    print(f'{"=" * 50}')
    print(f'  📡 数据源:          baostock（约 96 行/ETF）')
    print(f'  📊 训练ETF:          {", ".join(ETF_CODES)}')
    print(f'  💹 交易ETF:          {TRADING_ETF} ({ETF_DESCRIPTIONS[TRADING_ETF]})')
    print(f'  💰 初始资金:         {INITIAL_CAPITAL:,.0f} 元')
    print(f'  🎯 信号阈值:         {SIGNAL_THRESHOLD:.1%}')
    print(f'  📐 窗口大小:         {WINDOW_SIZE} 天')
    print(f'  🧠 模型:             {"简化版 LSTM" if USE_SIMPLE_MODEL else "完整版 LSTM-Transformer"}')
    print(f'  🔄 重训练间隔:       每 {MODEL_RETRAIN_INTERVAL} 个交易日')
    print(f'  🖥 训练并行:         {MAX_WORKERS} 线程')
    print(f'  💻 设备:             {DEVICE}')
    print(f'  💸 滑点:             {SLIPPAGE:.1%}')
    print(f'  💸 佣金:             {COMMISSION_RATE:.2%}')
    print(f'  🛑 止损:             {STOP_LOSS_PCT:.0%}')
    print(f'  📁 输出目录:         {OUTPUT_DIR}')
    print(f'  📁 模型目录:         {MODEL_DIR}')
    print(f'  📁 日志目录:         {LOG_DIR}')
    print(f'{"=" * 50}')
