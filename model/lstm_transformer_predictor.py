"""
LSTM-Transformer 预测模型 — ETF 版本

移植自项目 B（012_LSTM-Transformer_predict_stock_01），保留模型架构和训练流程，
移除贝叶斯优化（使用固定参数）、指数特征、市场特征依赖。

模型架构:
  输入 → 适配层 → 双向LSTM → Transformer编码器(带残差) → 全连接层 → 输出
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from config.etf_config import (
    DEVICE, PRODUCTION_MODEL_PARAMS, SIMPLE_MODEL_PARAMS,
    USE_SIMPLE_MODEL, MODEL_DIR, SEED,
    TRAIN_RATIO, VAL_RATIO, EARLY_STOPPING_PATIENCE,
    WINDOW_SIZE,
)
from model.feature_engineer import compute_features, create_sequences

# 设置随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# 数据集类
# ============================================================================

class StockDataset(Dataset):
    """PyTorch 数据集。"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# 简化版 LSTM 模型（轻量选项）
# ============================================================================

class SimpleLSTMModel(nn.Module):
    """
    简化版 LSTM 模型（无 Transformer）。

    当数据量较少或需要快速训练时使用。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.fc(last)


# ============================================================================
# 增强版 LSTM-Transformer 模型
# ============================================================================

class TransformerEncoderWithResidual(nn.Module):
    """带残差连接的 Transformer 编码器。"""

    def __init__(self, encoder_layer, num_layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        output = src
        for layer in self.layers:
            residual = output
            output = layer(output)
            output = output + residual
        return self.norm(output)


class EnhancedLSTMTransformerModel(nn.Module):
    """
    LSTM-Transformer 混合模型。

    架构: 输入 → 适配层 → 双向LSTM → Transformer编码器 → 全连接层 → 输出
    """

    def __init__(self, input_dim: int, params: dict = None):
        super().__init__()
        if params is None:
            params = DEFAULT_MODEL_PARAMS

        nhead = params['nhead']
        transformer_dim = params['transformer_dim']
        if transformer_dim % nhead != 0:
            transformer_dim -= transformer_dim % nhead

        # 1. 输入适配
        self.input_adapter = nn.Linear(input_dim, params['lstm_hidden'])

        # 2. 双向 LSTM
        self.lstm = nn.LSTM(
            input_size=params['lstm_hidden'],
            hidden_size=params['lstm_hidden'],
            num_layers=params['lstm_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=params['dropout'] if params['lstm_layers'] > 1 else 0,
        )

        # 3. LSTM → Transformer 适配
        lstm_out_dim = params['lstm_hidden'] * 2
        self.lstm_to_transformer = nn.Linear(lstm_out_dim, transformer_dim)

        # 4. Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=params['dropout'],
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = TransformerEncoderWithResidual(
            encoder_layer, params['num_transformer_layers'], transformer_dim,
        )

        # 5. 全连接层
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(transformer_dim, params['fc_hidden']))
        for _ in range(params['additional_fc_layers']):
            self.fc_layers.append(nn.Linear(params['fc_hidden'], params['fc_hidden']))
        self.output_layer = nn.Linear(params['fc_hidden'], 1)

        # 6. 激活和正则化
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(params['dropout'])
        self.layer_norm = nn.LayerNorm(transformer_dim)
        self.fc_norm = nn.LayerNorm(params['fc_hidden'])

        # 7. 位置编码（可学习）
        self.positional_encoding = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        x = self.input_adapter(x)
        lstm_out, _ = self.lstm(x)
        x = self.lstm_to_transformer(lstm_out)
        x = self.layer_norm(x)

        if self.positional_encoding is None or self.positional_encoding.size(1) != seq_len:
            self.positional_encoding = nn.Parameter(
                torch.zeros(1, seq_len, x.size(-1)), requires_grad=True,
            ).to(x.device)
        x = x + self.positional_encoding

        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # 取最后一个时间步

        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            x = self.activation(x)
            x = self.dropout(x)
            if i == 0:
                x = self.fc_norm(x)

        return self.output_layer(x)


# ============================================================================
# 预测器主类
# ============================================================================

class LSTMTransformerPredictor:
    """
    LSTM-Transformer 预测器。

    整合数据准备、模型训练、预测、保存/加载功能。
    """

    def __init__(self, use_simple: bool = False):
        """
        Parameters
        ----------
        use_simple : bool
            True=简化版LSTM（测试用），False=完整LSTM-Transformer（正式用）
        """
        self.device = DEVICE
        self.model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.use_simple = use_simple
        if use_simple:
            self.params = SIMPLE_MODEL_PARAMS.copy()
        else:
            self.params = PRODUCTION_MODEL_PARAMS.copy()

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    def prepare_data_from_df(
        self, df: pd.DataFrame,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray]]:
        """
        从 DataFrame 准备训练/验证/测试数据。

        Returns
        -------
        (X_train, y_train, X_val, y_val, X_test, y_test) or None
        """
        # 计算特征
        feat_df = compute_features(df)
        if feat_df is None or len(feat_df) < 40:
            return None

        # 创建序列
        X_seq, y_seq, self.feature_columns = create_sequences(feat_df)
        if len(X_seq) < 10:
            return None

        n = len(X_seq)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        X_train, X_val, X_test = X_seq[:train_end], X_seq[train_end:val_end], X_seq[val_end:]
        y_train, y_val, y_test = y_seq[:train_end], y_seq[train_end:val_end], y_seq[val_end:]

        # 标准化
        orig_shape = X_train.shape
        X_train_2d = X_train.reshape(-1, orig_shape[-1])
        X_scaled = self.feature_scaler.fit_transform(X_train_2d)
        X_train = X_scaled.reshape(orig_shape)

        if len(X_val) > 0:
            X_val_2d = X_val.reshape(-1, orig_shape[-1])
            X_val = self.feature_scaler.transform(X_val_2d).reshape(X_val.shape)

        if len(X_test) > 0:
            X_test_2d = X_test.reshape(-1, orig_shape[-1])
            X_test = self.feature_scaler.transform(X_test_2d).reshape(X_test.shape)

        # 目标标准化
        y_train = y_train.reshape(-1, 1)
        y_train = self.target_scaler.fit_transform(y_train).reshape(-1)

        if len(y_val) > 0:
            y_val = y_val.reshape(-1, 1)
            y_val = self.target_scaler.transform(y_val).reshape(-1)

        if len(y_test) > 0:
            y_test = y_test.reshape(-1, 1)
            y_test = self.target_scaler.transform(y_test).reshape(-1)

        return X_train, y_train, X_val, y_val, X_test, y_test

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, use_simple: Optional[bool] = None) -> bool:
        """
        训练模型。

        Parameters
        ----------
        df : pd.DataFrame
            ETF 日线数据
        use_simple : bool or None
            True=简化版LSTM（测试用），False=完整版，None=使用 self.use_simple

        Returns
        -------
        bool : 训练是否成功
        """
        if use_simple is None:
            use_simple = self.use_simple

        data = self.prepare_data_from_df(df)
        if data is None:
            print('  ⚠ 数据准备失败，无法训练')
            return False

        X_train, y_train, X_val, y_val, X_test, y_test = data
        input_dim = X_train.shape[2]

        if use_simple:
            self.model = SimpleLSTMModel(input_dim).to(self.device)
            print('  📦 使用简化版 LSTM 模型（测试用）')
        else:
            self.model = EnhancedLSTMTransformerModel(input_dim, self.params).to(self.device)
            print('  📦 使用增强版 LSTM-Transformer 模型（正式）')

        # 创建 DataLoader
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset, batch_size=self.params['batch_size'],
            shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.params['batch_size'],
            shuffle=False, num_workers=0,
        )

        # 训练
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
        best_val_loss = float('inf')
        patience = 0
        best_state = None

        for epoch in range(self.params['epochs']):
            # 训练
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    pred = self.model(batch_X)
                    val_loss += criterion(pred, batch_y).item()

            avg_val = val_loss / len(val_loader)
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = self.model.state_dict().copy()
                patience = 0
            else:
                patience += 1

            if (epoch + 1) % 20 == 0:
                print(f'  Epoch {epoch+1}/{self.params["epochs"]}  '
                      f'train_loss={train_loss/len(train_loader):.6f}  '
                      f'val_loss={avg_val:.6f}')

            if patience >= EARLY_STOPPING_PATIENCE:
                print(f'  早停 @ epoch {epoch+1}')
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # 测试集评估
        if len(X_test) > 0:
            self._evaluate(X_test, y_test)
        else:
            print(f'  最佳验证损失: {best_val_loss:.6f}')

        return True

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """在测试集上评估。"""
        self.model.eval()
        test_dataset = StockDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.params['batch_size'])

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                pred = self.model(batch_X).cpu().numpy()
                # 反标准化
                pred_orig = self.target_scaler.inverse_transform(pred)
                target_orig = self.target_scaler.inverse_transform(batch_y.numpy())
                all_preds.append(pred_orig)
                all_targets.append(target_orig)

        if all_preds:
            preds = np.concatenate(all_preds).ravel()
            targets = np.concatenate(all_targets).ravel()
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            mae = np.mean(np.abs(preds - targets))
            direction_acc = np.mean((preds > 0) == (targets > 0))

            print(f'  📊 测试集评估:')
            print(f'     RMSE: {rmse:.4f}%')
            print(f'     MAE:  {mae:.4f}%')
            print(f'     方向准确率: {direction_acc:.1%}')

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict_next_day(self, df: pd.DataFrame) -> Optional[float]:
        """
        预测下一交易日的涨跌幅（%）。

        Parameters
        ----------
        df : pd.DataFrame
            ETF 日线数据（最后 WINDOW_SIZE 行用于预测）

        Returns
        -------
        float or None
            预测的涨跌幅百分比，正=上涨，负=下跌
        """
        if self.model is None:
            return None

        feat_df = compute_features(df)
        if feat_df is None or len(feat_df) < WINDOW_SIZE + 5:
            return None

        # 取最后 WINDOW_SIZE 行
        last_data = feat_df.tail(WINDOW_SIZE).copy()
        if last_data.isnull().any().any():
            # 填充 NaN
            last_data = last_data.fillna(method='ffill').fillna(method='bfill').fillna(0)

        feature_cols = [c for c in self.feature_columns if c in last_data.columns]
        if len(feature_cols) < 5:
            return None

        values = last_data[feature_cols].values  # [window_size, n_features]
        # 标准化
        values_2d = values.reshape(-1, len(feature_cols))
        values_scaled = self.feature_scaler.transform(values_2d)
        X = values_scaled.reshape(1, len(values), -1)  # [1, window, features]

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred_scaled = self.model(X_tensor).cpu().numpy()  # [1, 1]
            pred = self.target_scaler.inverse_transform(pred_scaled)[0, 0]

        return float(pred)

    # ------------------------------------------------------------------
    # 保存 / 加载
    # ------------------------------------------------------------------

    def save(self, etf_code: str) -> str:
        """保存模型到文件。"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, f'etf_lstm_{etf_code}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'feature_columns': self.feature_columns,
            'params': self.params,
            'model_type': 'simple' if isinstance(self.model, SimpleLSTMModel) else 'full',
            'window_size': WINDOW_SIZE,
            'use_simple': self.use_simple,
        }, path)
        return path

    def load(self, etf_code: str) -> bool:
        """加载保存的模型。"""
        path = os.path.join(MODEL_DIR, f'etf_lstm_{etf_code}.pth')
        if not os.path.exists(path):
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.feature_scaler = checkpoint['feature_scaler']
            self.target_scaler = checkpoint['target_scaler']
            self.feature_columns = checkpoint.get('feature_columns', [])

            # 兼容旧版：没有 use_simple 字段时默认 full
            saved_use_simple = checkpoint.get('use_simple', False)
            self.use_simple = saved_use_simple
            self.params = checkpoint.get('params',
                                         SIMPLE_MODEL_PARAMS if saved_use_simple
                                         else PRODUCTION_MODEL_PARAMS)

            input_dim = len(self.feature_columns) if self.feature_columns else 30
            model_type = checkpoint.get('model_type', 'full')
            if model_type == 'simple':
                self.model = SimpleLSTMModel(input_dim).to(self.device)
            else:
                self.model = EnhancedLSTMTransformerModel(input_dim, self.params).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            return True

        except Exception as e:
            print(f'[模型] 加载 {etf_code} 失败: {e}')
            self.model = None
            return False

    def model_exists(self, etf_code: str) -> bool:
        path = os.path.join(MODEL_DIR, f'etf_lstm_{etf_code}.pth')
        return os.path.exists(path)
