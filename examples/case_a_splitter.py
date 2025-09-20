"""
案例A：高保真50/50分束器設計
Case A: High-fidelity 50/50 beam splitter design
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

from core.components import DesignParameters, DirectionalCoupler
from core.simulator import CircuitSimulator, create_simple_circuit
from optimization.bayesian_opt import optimize_design
from optimization.surrogate_model import SurrogateModel
from evaluation.metrics import quick_evaluate, MetricsCalculator
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

class SplitterDesignOptimizer:
    """50/50分束器設計最佳化器"""
    
    def __init__(self):
        self.simulator = create_simple_circuit(['directional_coupler'])
        self.simulator.set_quantum_simulator(n_modes=2, n_photons=1)
        self.optimization_history = []
        self.best_result = None
        
    def objective_function(self, params: Dict[str, float]) -> float:
        """
        目標函數：最大化50/50分束器的性能
        考慮傳輸效率、分束比準確性和製程容忍度
        """
        # 創建設計參數
        design_params = DesignParameters(
            coupling_length=params['coupling_length'],
            gap=params['gap'],
            waveguide_width=params['waveguide_width'],
            wavelength=1550e-9
        )
        
        try:
            # 執行模擬
            result = self.simulator.simulate_classical(design_params)
            
            # 評估性能
            evaluation = quick_evaluate(result, target_splitting_ratio=[0.5, 0.5])
            
            # 記錄歷史
            self.optimization_history.append({
                'params': params.copy(),
                'result': result,
                'evaluation': evaluation,
                'objective': evaluation['composite_score']
            })
            
            return evaluation['composite_score']
            
        except Exception as e:
            print(f"模擬錯誤: {e}")
            return 0.0
    
    def run_optimization(self, n_iterations: int = 50, method: str = 'bayesian') -> Dict:
        """
        執行設計最佳化
        
        Args:
            n_iterations: 最佳化迭代次數
            method: 最佳化方法 ('bayesian', 'surrogate')
        """
        print(f"=== 開始50/50分束器設計最佳化 ===")
        print(f"方法: {method}, 迭代次數: {n_iterations}")
        
        # 定義設計空間
        bounds = {
            'coupling_length': (5.0, 50.0),    # 耦合長度 (μm)
            'gap': (0.1, 1.0),                 # 間距 (μm)  
            'waveguide_width': (0.3, 0.7)      # 波導寬度 (μm)
        }
        
        start_time = time.time()
        
        if method == 'bayesian':
            # 貝葉斯最佳化
            best_params, best_value, history = optimize_design(
                self.objective_function,
                bounds,
                n_iterations=n_iterations,
                acquisition_func='ei',
                verbose=True
            )
            
        elif method == 'surrogate':
            # 使用代理模型的混合方法
            print("第一階段：生成初始訓練數據...")
            initial_samples = 30
            
            # 拉丁超立方採樣生成初始數據
            X_train = []
            y_train = []
            
            for i in range(initial_samples):
                params = {
                    'coupling_length': np.random.uniform(*bounds['coupling_length']),
                    'gap': np.random.uniform(*bounds['gap']),
                    'waveguide_width': np.random.uniform(*bounds['waveguide_width'])
                }
                objective_val = self.objective_function(params)
                X_train.append(params)
                y_train.append(objective_val)
                
                if (i + 1) % 10 == 0:
                    print(f"初始採樣進度: {i+1}/{initial_samples}")
            
            print("第二階段：訓練代理模型...")
            # 訓練代理模型
            surrogate = SurrogateModel(
                param_names=list(bounds.keys()),
                model_type='uncertainty'
            )
            surrogate.fit(X_train, y_train, epochs=100, verbose=False)
            
            print("第三階段：基於代理模型的貝葉斯最佳化...")
            # 基於代理模型的最佳化
            def surrogate_objective(params):
                pred, uncertainty = surrogate.predict(params)
                return pred
            
            remaining_iterations = n_iterations - initial_samples
            best_params, best_value, bo_history = optimize_design(
                self.objective_function,  # 仍使用真實函數進行最終評估
                bounds,
                n_iterations=remaining_iterations,
                acquisition_func='ei',
                verbose=True
            )
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        optimization_time = time.time() - start_time
        
        # 獲取最佳結果的詳細資訊
        best_design_params = DesignParameters(
            coupling_length=best_params['coupling_length'],
            gap=best_params['gap'],
            waveguide_width=best_params['waveguide_width'],
            wavelength=1550e-9
        )
        
        best_simulation = self.simulator.simulate_classical(best_design_params)
        best_evaluation = quick_evaluate(best_simulation, target_splitting_ratio=[0.5, 0.5])
        
        self.best_result = {
            'params': best_params,
            'simulation': best_simulation,
            'evaluation': best_evaluation,
            'optimization_time': optimization_time,
            'method': method,
            'n_iterations': n_iterations
        }
        
        return self.best_result
    
    def analyze_results(self):
        """分析最佳化結果"""
        if not self.best_result:
            print("尚未執行最佳化！")
            return
        
        print(f"\n=== 最佳化結果分析 ===")
        print(f"最佳化方法: {self.best_result['method']}")
        print(f"總迭代次數: {self.best_result['n_iterations']}")
        print(f"最佳化時間: {self.best_result['optimization_time']:.2f} 秒")
        
        print(f"\n最佳設計參數:")
        for param, value in self.best_result['params'].items():
            print(f"  {param}: {value:.4f}")
        
        print(f"\n性能指標:")
        summary = self.best_result['evaluation']['summary']
        for metric, value in summary.items():
            print(f"  {metric}: {value}")
        
        # 詳細分析
        sim = self.best_result['simulation']
        print(f"\n詳細分析:")
        print(f"  分束比: {sim.splitting_ratio[0]:.4f} / {sim.splitting_ratio[1]:.4f}")
        print(f"  理想比例偏差: {abs(sim.splitting_ratio[0] - 0.5):.4f}")
        print(f"  總功率損失: {1 - sum(sim.splitting_ratio):.4f}")
    
    def plot_optimization_history(self):
        """繪製最佳化歷史"""
        if not self.optimization_history:
            print("沒有最佳化歷史數據！")
            return
        
        # 提取數據
        iterations = range(len(self.optimization_history))
        objectives = [entry['objective'] for entry in self.optimization_history]
        best_so_far = [max(objectives[:i+1]) for i in range(len(objectives))]
        
        # 創建子圖
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 目標值歷史
        ax1.plot(iterations, objectives, 'b-', alpha=0.6, label='目標值')
        ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='目前最佳')
        ax1.set_xlabel('迭代次數')
        ax1.set_ylabel('目標值')
        ax1.set_title('最佳化收斂歷史')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 參數演化
        coupling_lengths = [entry['params']['coupling_length'] for entry in self.optimization_history]
        gaps = [entry['params']['gap'] for entry in self.optimization_history]
        widths = [entry['params']['waveguide_width'] for entry in self.optimization_history]
        
        ax2.plot(iterations, coupling_lengths, 'g-', label='耦合長度')
        ax2.set_xlabel('迭代次數')
        ax2.set_ylabel('耦合長度 (μm)')
        ax2.set_title('設計參數演化')
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(iterations, gaps, 'orange', label='間距')
        ax3.set_xlabel('迭代次數')
        ax3.set_ylabel('間距 (μm)')
        ax3.set_title('間距參數演化')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(iterations, widths, 'purple', label='波導寬度')
        ax4.set_xlabel('迭代次數')
        ax4.set_ylabel('波導寬度 (μm)')
        ax4.set_title('波導寬度演化')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_design_space_exploration(self):
        """繪製設計空間探索"""
        if len(self.optimization_history) < 10:
            print("數據點太少，無法繪製設計空間！")
            return
        
        # 提取數據
        coupling_lengths = [entry['params']['coupling_length'] for entry in self.optimization_history]
        gaps = [entry['params']['gap'] for entry in self.optimization_history]
        objectives = [entry['objective'] for entry in self.optimization_history]
        
        # 3D散點圖
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(coupling_lengths, gaps, objectives, 
                           c=objectives, cmap='viridis', s=50, alpha=0.7)
        
        # 標記最佳點
        best_idx = np.argmax(objectives)
        ax.scatter([coupling_lengths[best_idx]], [gaps[best_idx]], [objectives[best_idx]], 
                  c='red', s=200, marker='*', label='最佳解')
        
        ax.set_xlabel('耦合長度 (μm)')
        ax.set_ylabel('間距 (μm)')
        ax.set_zlabel('目標值')
        ax.set_title('設計空間探索')
        
        plt.colorbar(scatter)
        ax.legend()
        plt.show()

def main():
    """主函數：執行50/50分束器設計案例"""
    print("50/50分束器設計最佳化案例")
    print("=" * 50)
    
    # 創建最佳化器
    optimizer = SplitterDesignOptimizer()
    
    # 執行貝葉斯最佳化
    print("\n1. 執行貝葉斯最佳化...")
    result_bo = optimizer.run_optimization(n_iterations=40, method='bayesian')
    optimizer.analyze_results()
    
    # 繪製結果
    print("\n2. 繪製最佳化歷史...")
    optimizer.plot_optimization_history()
    
    print("\n3. 繪製設計空間探索...")
    optimizer.plot_design_space_exploration()
    
    # 測試代理模型方法
    print("\n4. 測試代理模型方法...")
    optimizer_surrogate = SplitterDesignOptimizer()
    result_surrogate = optimizer_surrogate.run_optimization(n_iterations=40, method='surrogate')
    
    # 比較結果
    print(f"\n=== 方法比較 ===")
    print(f"貝葉斯最佳化:")
    print(f"  最佳目標值: {result_bo['evaluation']['composite_score']:.4f}")
    print(f"  最佳化時間: {result_bo['optimization_time']:.2f} 秒")
    
    print(f"代理模型方法:")
    print(f"  最佳目標值: {result_surrogate['evaluation']['composite_score']:.4f}")
    print(f"  最佳化時間: {result_surrogate['optimization_time']:.2f} 秒")

if __name__ == "__main__":
    main()