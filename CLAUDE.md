# 016_etf_lstm_predict

沪深300 ETF LSTM-Transformer 预测模拟盘系统。用深度学习模型预测 ETF 次日涨跌幅，根据预测信号自动交易，每日通过 WxPusher 推送模拟盘日报。

## 核心思路

> **用 LSTM-Transformer 模型预测 ETF 次日方向，以交易成本为阈值生成买卖信号**

### 系统架构

```
每日 cron 21:00 触发
  → 交易日判断（baostock API）
  → ETF 数据加载（baostock, 45 秒超时保护）
  → 模型预测次日涨跌幅（已有模型）
  → 阈值判断 → 生成买卖信号（>0.36% 买入, <-0.36% 卖出）
  → 执行昨日待处理订单
  → 计算策略 vs 基准（buy-and-hold）收益率
  → WxPusher 推送日报
```

### 信号逻辑

```
预测次日涨幅 > 0.36%  → 买入（下一交易日开盘价）
预测次日跌幅 < -0.36% → 卖出（下一交易日开盘价）
其余                  → 持有不动
```

- 0.36% = 滑点(0.3%) + 佣金(0.05%) + 预估冲击成本(0.01%)
- 确保交易覆盖成本后才有正向期望

### 四只沪深300 ETF

| 代码 | 名称 | 交易所 |
|------|------|--------|
| 510300 | 华泰柏瑞沪深300ETF | 上海 |
| 510310 | 易方达沪深300ETF | 上海 |
| 510330 | 华夏沪深300ETF | 上海 |
| 159919 | 嘉实沪深300ETF | 深圳 |

## 项目结构

```
016_etf_lstm_predict/
├── run_daily.py                    # 唯一入口（cron 调用）
├── config/
│   └── etf_config.py               # 全局配置
├── core/
│   ├── log_utils.py                # 日志（ANSI 彩色输出）
│   ├── hs300_utils.py              # 交易日判断 + baostock 数据获取（含45s超时保护）
│   ├── notification.py             # WxPusher 推送
│   ├── state_manager.py            # JSON 状态持久化（原子写入）
│   └── simulator.py                # 模拟盘引擎
├── model/
│   ├── __init__.py
│   ├── feature_engineer.py         # 特征工程（精简版，适配ETF数据量）
│   └── lstm_transformer_predictor.py # LSTM-Transformer + SimpleLSTM 预测模型
├── output/
│   ├── state.json                  # 状态文件（运行时创建）
│   ├── models/                     # 模型文件 (.pth)
│   └── .run_etf.lock              # PID 锁文件（防重复运行）
└── logs/                           # 运行日志
```

## 快速上手

### 1. 环境准备

```bash
source activate zhulei
cd /public/home/hpc/zhulei/superman/quant/code/016_etf_lstm_predict
```

### 2. 首次训练模型

```bash
# 训练全部 4 只 ETF 的 LSTM 模型（约 8~12 分钟）
python run_daily.py --train
```

### 3. 每日模拟盘

```bash
# 执行模拟盘（交易日检查 + 预测 + 交易 + 推送）
python run_daily.py

# Dry-run 模式（不修改状态、不推送）
python run_daily.py --dry-run
```

## Cron 配置

```bash
# 每日模拟盘（交易日 21:00）
0 21 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/016_etf_lstm_predict && \
  /home/zhulei/anaconda3/envs/zhulei/bin/python run_daily.py \
  >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

## 模型架构

本系统的模型移植自 `012_LSTM-Transformer_predict_stock_01`，保留其核心设计：
- 双向 LSTM 捕捉时序依赖
- Transformer 编码器（带残差连接）捕捉长期模式
- 全连接输出层预测次日涨跌幅

为适配 ETF 数据量（baostock 约 96 行），做了以下精简：
- 默认使用 SimpleLSTMModel（更快，约 3 分钟/ETF）
- 特征从 ~80 维精简到 ~32 维（去掉指数特征避免循环引用）
- 窗口大小 30 天
- 早停机制防止过拟合

## 关键技术细节

- **消除 look-ahead bias**：信号基于今日收盘数据，明日开盘执行
- **baostock 45s 超时保护**：防止 API 挂起导致脚本永久卡死
- **PID 锁文件**：启动时自动清理残留进程，退出时自动清理
- **ETF 免印花税**：交易成本仅为佣金 + 滑点
- **原子写入**：state.json 先写 .tmp 再 rename
