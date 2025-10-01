"""
案例A-量子版：高保真50/50分束器設計與量子模擬分析
Case A - Quantum Version: High-fidelity 50/50 beam splitter design with quantum simulation analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import time

from core.components import DesignParameters
from core.simulator import CircuitSimulator, create_simple_circuit
from optimization.bayesian_opt import optimize_design
from evaluation.metrics import quick_evaluate
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

class SplitterDesignOptimizer:
    """50/50分束器設計最佳化器"""
    
    def __init__(self):
        self.simulator = create_simple_circuit(['directional_coupler'])
        # 確保設置了量子模擬器以備後用
        self.simulator.set_quantum_simulator(n_modes=2, n_photons=1)
        self.optimization_history = []
        self.best_result = None
        
    def objective_function(self, params: Dict[str, float]) -> float:
        """目標函數：最大化50/50分束器的性能"""
        design_params = DesignParameters(
            coupling_length=params['coupling_length'],
            gap=params['gap'],
            waveguide_width=params['waveguide_width'],
            wavelength=1550e-9
        )
        
        try:
            result = self.simulator.simulate_classical(design_params)
            evaluation = quick_evaluate(result, target_splitting_ratio=[0.5, 0.5])
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
    
    def run_optimization(self, n_iterations: int = 50) -> Dict:
        """執行設計最佳化"""
        print(f"=== 開始50/50分束器設計最佳化 ===")
        print(f"迭代次數: {n_iterations}")
        
        bounds = {
            'coupling_length': (5.0, 50.0),
            'gap': (0.05, 1.0),
            'waveguide_width': (0.25, 0.7)
        }
        
        start_time = time.time()
        
        best_params, best_value, history = optimize_design(
            self.objective_function,
            bounds,
            n_iterations=n_iterations,
            acquisition_func='ei',
            verbose=True
        )
        
        optimization_time = time.time() - start_time
        
        best_design_params = DesignParameters(
            coupling_length=best_params['coupling_length'],
            gap=best_params['gap'],
            waveguide_width=best_params['waveguide_width']
        )
        
        best_simulation = self.simulator.simulate_classical(best_design_params)
        best_evaluation = quick_evaluate(best_simulation, target_splitting_ratio=[0.5, 0.5])
        
        self.best_result = {
            'params': best_params,
            'simulation': best_simulation,
            'evaluation': best_evaluation,
            'optimization_time': optimization_time,
            'n_iterations': n_iterations
        }
        
        return self.best_result
    
    def analyze_classical_results(self):
        """分析傳統模擬的最佳化結果"""
        if not self.best_result:
            print("尚未執行最佳化！")
            return
        
        print(f"\n=== 經典模擬結果分析 ===")
        print(f"總迭代次數: {self.best_result['n_iterations']}")
        print(f"最佳化時間: {self.best_result['optimization_time']:.2f} 秒")
        
        print(f"\n最佳設計參數:")
        for param, value in self.best_result['params'].items():
            print(f"  {param}: {value:.4f}")
        
        print(f"\n性能指標 (基於經典模擬):")
        summary = self.best_result['evaluation']['summary']
        for metric, value in summary.items():
            print(f"  {metric}: {value}")
            
    def analyze_quantum_simulation(self):
        """對最佳參數執行並分析量子模擬"""
        if not self.best_result:
            print("尚未執行最佳化，無法進行量子模擬分析！")
            return
            
        print(f"\n=== 量子模擬分析 ===")
        print("使用找到的最佳參數，模擬單光子輸入 |1,0> 的行為...")
        
        best_params_dict = self.best_result['params']
        best_design_params = DesignParameters(
            coupling_length=best_params_dict['coupling_length'],
            gap=best_params_dict['gap'],
            waveguide_width=best_params_dict['waveguide_width']
        )
        
        # 執行量子模擬
        quantum_result = self.simulator.simulate_quantum(best_design_params)
        
        # 顯示量子模擬結果
        photon_numbers = quantum_result['photon_numbers']
        quantum_fidelity = quantum_result['quantum_fidelity']
        
        print(f"\n⚛️ 量子模擬結果:")
        print(f"  - 輸入態: |1,0> (單光子在第一個波導)")
        print(f"  - 輸出光子分佈 (期望值):")
        print(f"    - 端口 1: {photon_numbers[0]:.4f}")
        print(f"    - 端口 2: {photon_numbers[1]:.4f}")
        print(f"  - 量子保真度 (與理想 |1,0> -> (|1,0>+i|0,1>)/sqrt(2) 比較): {quantum_fidelity:.4f}")
        
        # 繪製光子分佈
        fig, ax = plt.subplots(figsize=(8, 6))
        ports = ['端口 1', '端口 2']
        ax.bar(ports, photon_numbers, color=['skyblue', 'salmon'])
        ax.set_ylabel('光子數期望值')
        ax.set_title('最佳化分束器的單光子輸出分佈')
        ax.set_ylim(0, 1.0)
        for i, v in enumerate(photon_numbers):
            ax.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
        plt.show()


def main():
    """主函數：執行50/50分束器設計與量子分析"""
    print("="*60)
    print("案例A-量子版：50/50分束器設計與量子模擬分析")
    print("="*60)
    
    # 1. 創建並執行最佳化
    optimizer = SplitterDesignOptimizer()
    optimizer.run_optimization(n_iterations=50)
    
    # 2. 分析經典模擬結果
    optimizer.analyze_classical_results()
    
    # 3. 執行並分析量子模擬
    optimizer.analyze_quantum_simulation()

if __name__ == "__main__":
    main()
