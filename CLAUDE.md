# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 016_etf_lstm_predict

沪深300 ETF LSTM-Transformer 预测模拟盘系统。用深度学习模型预测 ETF 次日涨跌幅，以 0.8% 为阈值生成买卖信号，每日通过 WxPusher 推送模拟盘日报。

## 核心思路

> **用 LSTM 模型预测 ETF 次日方向，以 0.8% 阈值过滤假信号**

### 信号逻辑

```
预测涨幅 > +0.80% → 🟢 买入（下一交易日开盘价执行）
预测跌幅 < -0.80% → 🔴 卖出（下一交易日开盘价执行）
其余              → ⚪ 持有不动
```

## 常用命令

```bash
# 每日模拟盘（cron 用）
python run_daily.py

# 强制重训练所有模型
python run_daily.py --train

# Dry-run 不修改状态
python run_daily.py --dry-run

# 单独执行模拟盘阶段
python run_daily.py --phase 4
```

## Cron 配置

```bash
# 每日 21:10 运行（与项目 A 015_indicator_scanner 的 21:00 错开）
10 21 * * 1-5 cd /path/to/016_etf_lstm_predict && \
  /home/zhulei/anaconda3/envs/zhulei/bin/python run_daily.py \
  >> logs/daily_$(date +\%Y\%m\%d).log 2>&1
```

## 项目结构

```
016_etf_lstm_predict/
├── run_daily.py                    # 唯一入口
├── config/
│   └── etf_config.py               # 全局配置
├── core/
│   ├── log_utils.py                # 日志（ANSI 彩色）
│   ├── hs300_utils.py              # baostock 交易日+数据获取（45s超时）
│   ├── notification.py             # WxPusher 推送
│   ├── state_manager.py            # JSON 状态持久化
│   └── simulator.py                # 模拟盘引擎
├── model/
│   ├── feature_engineer.py         # 特征工程（~30 维技术指标）
│   └── lstm_transformer_predictor.py # LSTM-Transformer + SimpleLSTM
├── output/
│   ├── state.json                  # 状态文件
│   └── models/                     # .pth 模型文件
└── logs/
```

## 模型架构

`model/lstm_transformer_predictor.py` 包含两种模型：

| 模型 | 结构 | 适用场景 |
|------|------|---------|
| **SimpleLSTMModel** | LSTM(32) → FC(16) → 输出 | ✅ 默认（数据少，防过拟合） |
| **EnhancedLSTMTransformerModel** | 适配层 → 双向LSTM → Transformer编码器(残差) → FC → 输出 | 数据充足时可用 |

训练流程：特征计算 → 序列化(窗口20) → 标准化 → DataLoader → 训练(早停) → 保存

## 关键技术细节

- **数据源**：baostock，ETF 仅回溯 ~5 个月（96 行），窗口 20 天得 ~50 样本
- **训练**：串行加载 → ThreadPoolExecutor 4 线程并行，4 只 ETF 约 22 秒
- **重训练**：每 5 个交易日自动触发，`PRODUCTION_MODEL_PARAMS`（生产）或 `SIMPLE_MODEL_PARAMS`（测试） 
- **防 look-ahead bias**：信号基于今日收盘，明日开盘执行
- **防重复运行**：PID 锁文件 + `/proc` 进程匹配，退出自动清理
- **baostock 超时**：45 秒 `signal.alarm`，防止 API 挂起
- **ETF 免印花税**：交易成本仅为佣金(万5) + 滑点(0.3%)
- **配置中心**：`config/etf_config.py` 统一管理所有参数（阈值、窗口、模型参数、WxPusher）

## 与相邻项目的关系

| 项目 | 关系 |
|------|------|
| `015_indicator_scanner` | 共享 core/ 模块设计理念和技术指标参数 |
| `012_LSTM-Transformer_predict_stock_01` | LSTM-Transformer 模型代码移植来源 |
