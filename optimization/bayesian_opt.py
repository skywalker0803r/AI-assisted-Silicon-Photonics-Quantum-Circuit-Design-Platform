"""
Bayesian Optimization for silicon photonics design
貝葉斯最佳化模組
"""
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

class BayesianOptimizer:
    """貝葉斯最佳化器"""
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]], 
                 acquisition_func: str = 'ei', 
                 n_random_starts: int = 10):
        """
        初始化貝葉斯最佳化器
        
        Args:
            bounds: 參數邊界 {'param_name': (min, max)}
            acquisition_func: 獲取函數類型 ('ei', 'pi', 'ucb')
            n_random_starts: 隨機初始採樣點數量
        """
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.param_bounds = np.array([bounds[name] for name in self.param_names])
        self.acquisition_func = acquisition_func
        self.n_random_starts = n_random_starts
        
        # 初始化高斯過程
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        # 儲存歷史數據
        self.X_observed = []
        self.y_observed = []
        self.iteration = 0
        
    def _normalize_params(self, params: np.ndarray) -> np.ndarray:
        """將參數正規化到[0,1]"""
        return (params - self.param_bounds[:, 0]) / (self.param_bounds[:, 1] - self.param_bounds[:, 0])
    
    def _denormalize_params(self, params_norm: np.ndarray) -> np.ndarray:
        """將正規化參數轉換回原始範圍"""
        return params_norm * (self.param_bounds[:, 1] - self.param_bounds[:, 0]) + self.param_bounds[:, 0]
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, float]:
        """將參數陣列轉換為字典"""
        return {name: float(params[i]) for i, name in enumerate(self.param_names)}
    
    def _dict_to_params(self, params_dict: Dict[str, float]) -> np.ndarray:
        """將參數字典轉換為陣列"""
        return np.array([params_dict[name] for name in self.param_names])
    
    def _latin_hypercube_sampling(self, n_samples: int) -> np.ndarray:
        """拉丁超立方採樣"""
        n_params = len(self.param_names)
        samples = np.zeros((n_samples, n_params))
        
        for i in range(n_params):
            samples[:, i] = np.random.uniform(0, 1, n_samples)
            
        # 打亂每個維度
        for i in range(n_params):
            np.random.shuffle(samples[:, i])
            
        return self._denormalize_params(samples)
    
    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """期望改善獲取函數"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        
        # 避免除零
        sigma = sigma.reshape(-1, 1)
        sigma = np.maximum(sigma, 1e-9)
        
        if len(self.y_observed) > 0:
            mu_sample_opt = np.max(self.y_observed)
            
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
                ei[sigma == 0.0] = 0.0
                
        else:
            ei = sigma
            
        return ei.flatten()
    
    def _probability_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """改善概率獲取函數"""
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if len(self.y_observed) > 0:
            mu_sample_opt = np.max(self.y_observed)
            sigma = np.maximum(sigma, 1e-9)
            
            Z = (mu - mu_sample_opt - xi) / sigma
            pi = self._normal_cdf(Z)
        else:
            pi = np.ones_like(mu)
            
        return pi
    
    def _upper_confidence_bound(self, X: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """上信賴界獲取函數"""
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma
    
    def _normal_cdf(self, x):
        """標準正態分佈累積分佈函數"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _normal_pdf(self, x):
        """標準正態分佈概率密度函數"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """獲取函數"""
        if self.acquisition_func == 'ei':
            return self._expected_improvement(X)
        elif self.acquisition_func == 'pi':
            return self._probability_improvement(X)
        elif self.acquisition_func == 'ucb':
            return self._upper_confidence_bound(X)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_func}")
    
    def _optimize_acquisition(self) -> np.ndarray:
        """最佳化獲取函數以找到下一個採樣點"""
        def objective(x):
            x_norm = self._normalize_params(x.reshape(1, -1))
            return -self._acquisition_function(x_norm)[0]
        
        # 多次隨機起點最佳化
        best_x = None
        best_val = float('inf')
        
        for _ in range(10):
            x0 = np.random.uniform(
                self.param_bounds[:, 0], 
                self.param_bounds[:, 1]
            )
            
            result = minimize(
                objective, 
                x0, 
                bounds=[(low, high) for low, high in self.param_bounds],
                method='L-BFGS-B'
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
                
        return best_x
    
    def suggest_next(self) -> Dict[str, float]:
        """建議下一個採樣點"""
        if len(self.X_observed) < self.n_random_starts:
            # 隨機採樣階段
            next_params = np.random.uniform(
                self.param_bounds[:, 0], 
                self.param_bounds[:, 1]
            )
        else:
            # 貝葉斯最佳化階段
            if len(self.X_observed) > 0:
                X_norm = np.array([self._normalize_params(x) for x in self.X_observed])
                self.gp.fit(X_norm, self.y_observed)
            
            next_params = self._optimize_acquisition()
        
        return self._params_to_dict(next_params)
    
    def tell(self, params: Dict[str, float], objective_value: float):
        """告知最佳化器觀察結果"""
        params_array = self._dict_to_params(params)
        self.X_observed.append(params_array)
        self.y_observed.append(objective_value)
        self.iteration += 1
    
    def get_best(self) -> Tuple[Dict[str, float], float]:
        """獲取目前最佳解"""
        if not self.y_observed:
            raise ValueError("No observations yet")
            
        best_idx = np.argmax(self.y_observed)
        best_params = self._params_to_dict(self.X_observed[best_idx])
        best_value = self.y_observed[best_idx]
        
        return best_params, best_value
    
    def predict(self, params: Dict[str, float]) -> Tuple[float, float]:
        """預測給定參數的目標值和不確定度"""
        if len(self.X_observed) == 0:
            return 0.0, 1.0
            
        params_array = self._dict_to_params(params)
        params_norm = self._normalize_params(params_array.reshape(1, -1))
        
        X_norm = np.array([self._normalize_params(x) for x in self.X_observed])
        self.gp.fit(X_norm, self.y_observed)
        
        mu, sigma = self.gp.predict(params_norm, return_std=True)
        return float(mu[0]), float(sigma[0])

# 便利函數
def optimize_design(objective_func: Callable, 
                   bounds: Dict[str, Tuple[float, float]], 
                   n_iterations: int = 50,
                   acquisition_func: str = 'ei',
                   verbose: bool = True) -> Tuple[Dict[str, float], float, List]:
    """
    執行貝葉斯最佳化設計
    
    Args:
        objective_func: 目標函數，接受參數字典，返回目標值
        bounds: 參數邊界
        n_iterations: 最佳化迭代次數
        acquisition_func: 獲取函數類型
        verbose: 是否顯示進度
    
    Returns:
        最佳參數、最佳值、歷史記錄
    """
    optimizer = BayesianOptimizer(bounds, acquisition_func)
    history = []
    
    for i in range(n_iterations):
        # 獲取下一個建議點
        next_params = optimizer.suggest_next()
        
        # 評估目標函數
        objective_value = objective_func(next_params)
        
        # 告知結果
        optimizer.tell(next_params, objective_value)
        
        # 記錄歷史
        history.append({
            'iteration': i,
            'params': next_params.copy(),
            'objective': objective_value
        })
        
        if verbose and (i + 1) % 10 == 0:
            best_params, best_value = optimizer.get_best()
            print(f"Iteration {i+1}/{n_iterations}, Best: {best_value:.4f}")
    
    best_params, best_value = optimizer.get_best()
    
    if verbose:
        print(f"\n=== 最佳化完成 ===")
        print(f"最佳目標值: {best_value:.4f}")
        print("最佳參數:")
        for name, value in best_params.items():
            print(f"  {name}: {value:.4f}")
    
    return best_params, best_value, history

# 測試範例
if __name__ == "__main__":
    # 測試函數：最小化 Rosenbrock 函數
    def rosenbrock(params):
        x, y = params['x'], params['y']
        return -(100 * (y - x**2)**2 + (1 - x)**2)  # 負號因為我們要最大化
    
    bounds = {
        'x': (-2.0, 2.0),
        'y': (-2.0, 2.0)
    }
    
    best_params, best_value, history = optimize_design(
        rosenbrock, bounds, n_iterations=30, verbose=True
    )
    
    print(f"\n真實最佳解: x=1, y=1")
    print(f"找到的解: x={best_params['x']:.3f}, y={best_params['y']:.3f}")