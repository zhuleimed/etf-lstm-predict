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

选 0.8% 作为阈值的原因：ETF 日波动通常在 0.5%~1.5%，0.8% 可过滤大部分随机波动，确保交易信号有实际价值。

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
│   └── etf_config.py               # 全局配置（阈值/窗口/模型参数）
├── core/
│   ├── log_utils.py                # 日志（ANSI 彩色输出）
│   ├── hs300_utils.py              # 交易日判断 + baostock 数据获取（45s超时保护）
│   ├── notification.py             # WxPusher 推送
│   ├── state_manager.py            # JSON 状态持久化（原子写入）
│   └── simulator.py                # 模拟盘引擎
├── model/
│   ├── __init__.py
│   ├── feature_engineer.py         # 特征工程（32 维技术指标）
│   └── lstm_transformer_predictor.py # LSTM-Transformer + SimpleLSTM
├── output/
│   ├── state.json
│   ├── models/                     # 模型文件 (.pth)
│   └── .run_etf.lock
└── logs/
```

## 关键技术细节

### 数据源：baostock（约 96 行）
- baostock 对 ETF 只回溯约 5 个月（2026-01 起），每只约 96 行
- 窗口大小 20 天 → 可得约 50 个样本
- 小数据量下推荐简化版 LSTM（过拟合风险更低）
- 所有 API 调用设 45 秒超时保护

### 训练：串行加载 + 线程池并行训练
```
串行加载: ETF1→ETF2→ETF3→ETF4 (baostock, 约 8 秒)
        ↓
线程池: ┌─ ETF1 ─┐  (4 线程并行)
        ├─ ETF2 ─┤
        ├─ ETF3 ─┤
        └─ ETF4 ─┘
        ↓
总耗时 ≈ 耗时最长的单只 ETF (约 20~30 秒)
```

### 其他
- 消除 look-ahead bias：信号基于今日收盘，明日开盘执行
- 防重复运行：PID 锁文件 + `/proc` 进程匹配
- ETF 免印花税，交易成本仅为佣金+滑点
- 模型每 5 个交易日自动重训练（并行 4 线程）
