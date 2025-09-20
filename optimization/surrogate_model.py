"""
Neural Network Surrogate Models for fast design space exploration
神經網路代理模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DesignDataset(Dataset):
    """設計參數與目標值的數據集"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.reshape(-1, 1))
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SurrogateNet(nn.Module):
    """代理神經網路模型"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16], 
                 dropout_rate: float = 0.1):
        super(SurrogateNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        # 輸出層
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class UncertaintyNet(nn.Module):
    """帶不確定度估計的神經網路"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super(UncertaintyNet, self).__init__()
        
        # 共享特徵提取器
        feature_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 均值和方差分支
        final_dim = hidden_dims[-1]
        self.mean_head = nn.Sequential(
            nn.Linear(prev_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1)
        )
        
        self.var_head = nn.Sequential(
            nn.Linear(prev_dim, final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, 1),
            nn.Softplus()  # 確保方差為正
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        var = self.var_head(features)
        return mean, var

class SurrogateModel:
    """代理模型包裝器"""
    
    def __init__(self, param_names: List[str], model_type: str = 'standard',
                 hidden_dims: List[int] = [64, 32, 16]):
        """
        初始化代理模型
        
        Args:
            param_names: 參數名稱列表
            model_type: 模型類型 ('standard', 'uncertainty')
            hidden_dims: 隱藏層維度
        """
        self.param_names = param_names
        self.input_dim = len(param_names)
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        
        # 初始化模型
        if model_type == 'standard':
            self.model = SurrogateNet(self.input_dim, hidden_dims)
        elif model_type == 'uncertainty':
            self.model = UncertaintyNet(self.input_dim, hidden_dims)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 數據標準化器
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # 訓練歷史
        self.training_history = []
        self.is_fitted = False
        
    def _params_dict_to_array(self, params_list: List[Dict[str, float]]) -> np.ndarray:
        """將參數字典列表轉換為陣列"""
        return np.array([[params[name] for name in self.param_names] 
                        for params in params_list])
    
    def _params_array_to_dict(self, params_array: np.ndarray) -> Dict[str, float]:
        """將參數陣列轉換為字典"""
        return {name: float(params_array[i]) for i, name in enumerate(self.param_names)}
    
    def fit(self, X_data: List[Dict[str, float]], y_data: List[float],
            epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
            validation_split: float = 0.2, verbose: bool = True):
        """
        訓練代理模型
        
        Args:
            X_data: 輸入參數列表
            y_data: 目標值列表
            epochs: 訓練周期
            batch_size: 批次大小
            learning_rate: 學習率
            validation_split: 驗證集比例
            verbose: 是否顯示訓練進度
        """
        # 準備數據
        X = self._params_dict_to_array(X_data)
        y = np.array(y_data)
        
        # 標準化
        X_scaled = self.x_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 分割訓練/驗證集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        # 創建數據載入器
        train_dataset = DesignDataset(X_train, y_train)
        val_dataset = DesignDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 優化器和損失函數
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if self.model_type == 'standard':
            criterion = nn.MSELoss()
        else:  # uncertainty model
            def nll_loss(mean, var, target):
                return torch.mean(0.5 * torch.log(var) + 0.5 * (target - mean)**2 / var)
            criterion = nll_loss
        
        # 訓練循環
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                if self.model_type == 'standard':
                    output = self.model(batch_x)
                    loss = criterion(output, batch_y)
                else:
                    mean, var = self.model(batch_x)
                    loss = criterion(mean, var, batch_y)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 驗證階段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    if self.model_type == 'standard':
                        output = self.model(batch_x)
                        loss = criterion(output, batch_y)
                    else:
                        mean, var = self.model(batch_x)
                        loss = criterion(mean, var, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        self.is_fitted = True
        
        if verbose:
            print(f"訓練完成！最終驗證損失: {val_losses[-1]:.6f}")
    
    def predict(self, params: Dict[str, float]) -> Tuple[float, Optional[float]]:
        """
        預測給定參數的目標值
        
        Args:
            params: 輸入參數字典
            
        Returns:
            預測值和不確定度（如果是不確定度模型）
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        # 準備輸入
        X = np.array([params[name] for name in self.param_names]).reshape(1, -1)
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'standard':
                output = self.model(X_tensor)
                pred_scaled = output.numpy()[0, 0]
                uncertainty = None
            else:
                mean, var = self.model(X_tensor)
                pred_scaled = mean.numpy()[0, 0]
                uncertainty_scaled = np.sqrt(var.numpy()[0, 0])
                uncertainty = float(self.y_scaler.scale_[0] * uncertainty_scaled)
        
        # 反標準化
        prediction = float(self.y_scaler.inverse_transform([[pred_scaled]])[0, 0])
        
        return prediction, uncertainty
    
    def predict_batch(self, params_list: List[Dict[str, float]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """批次預測"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        X = self._params_dict_to_array(params_list)
        X_scaled = self.x_scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'standard':
                outputs = self.model(X_tensor)
                pred_scaled = outputs.numpy().flatten()
                uncertainties = None
            else:
                means, vars = self.model(X_tensor)
                pred_scaled = means.numpy().flatten()
                uncertainties_scaled = np.sqrt(vars.numpy().flatten())
                uncertainties = self.y_scaler.scale_[0] * uncertainties_scaled
        
        predictions = self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        return predictions, uncertainties
    
    def plot_training_history(self):
        """繪製訓練歷史"""
        if not self.training_history:
            print("No training history available")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['train_losses'], label='Training Loss')
        plt.plot(self.training_history['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'param_names': self.param_names,
            'model_type': self.model_type,
            'hidden_dims': self.hidden_dims,
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load(self, filepath: str):
        """加載模型"""
        checkpoint = torch.load(filepath)
        
        self.param_names = checkpoint['param_names']
        self.model_type = checkpoint['model_type']
        self.hidden_dims = checkpoint['hidden_dims']
        self.x_scaler = checkpoint['x_scaler']
        self.y_scaler = checkpoint['y_scaler']
        self.training_history = checkpoint['training_history']
        self.is_fitted = checkpoint['is_fitted']
        
        # 重建模型
        if self.model_type == 'standard':
            self.model = SurrogateNet(len(self.param_names), self.hidden_dims)
        else:
            self.model = UncertaintyNet(len(self.param_names), self.hidden_dims)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])

# 測試範例
if __name__ == "__main__":
    # 生成測試數據
    def test_function(params):
        x, y = params['x'], params['y']
        return x**2 + y**2 + 0.1 * np.random.randn()
    
    # 生成訓練數據
    n_samples = 200
    param_names = ['x', 'y']
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        params = {'x': np.random.uniform(-2, 2), 'y': np.random.uniform(-2, 2)}
        X_data.append(params)
        y_data.append(test_function(params))
    
    # 訓練標準模型
    print("=== 訓練標準代理模型 ===")
    surrogate = SurrogateModel(param_names, model_type='standard')
    surrogate.fit(X_data, y_data, epochs=100, verbose=True)
    
    # 測試預測
    test_params = {'x': 1.0, 'y': 1.0}
    pred, _ = surrogate.predict(test_params)
    true_val = test_function(test_params)
    print(f"\n測試點 {test_params}")
    print(f"真實值: {true_val:.4f}")
    print(f"預測值: {pred:.4f}")
    print(f"誤差: {abs(pred - true_val):.4f}")
    
    # 訓練不確定度模型
    print("\n=== 訓練不確定度代理模型 ===")
    uncertainty_surrogate = SurrogateModel(param_names, model_type='uncertainty')
    uncertainty_surrogate.fit(X_data, y_data, epochs=100, verbose=True)
    
    # 測試不確定度預測
    pred_unc, uncertainty = uncertainty_surrogate.predict(test_params)
    print(f"\n不確定度模型預測:")
    print(f"預測值: {pred_unc:.4f} ± {uncertainty:.4f}")
    print(f"真實值: {true_val:.4f}")