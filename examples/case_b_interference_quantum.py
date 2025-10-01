"""
案例B-量子版：三輸入干涉電路設計與量子模擬分析
Case B - Quantum Version: Three-input interference circuit design with quantum simulation analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

from core.components import DesignParameters, ThreePortInterferometer
from core.simulator import CircuitSimulator
from optimization.bayesian_opt import optimize_design
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

class BosonSamplingOptimizer:
    """Boson Sampling干涉電路最佳化器"""
    
    def __init__(self):
        self.setup_circuit()
        self.optimization_history = []
        self.best_result = None

    def setup_circuit(self):
        """設置三端口干涉電路"""
        self.simulator = CircuitSimulator()
        self.interferometer = ThreePortInterferometer()
        self.simulator.add_component(self.interferometer)
        # 設置量子模擬器以備後用，2個光子在3個模式中
        self.simulator.set_quantum_simulator(n_modes=3, n_photons=2)
    
    def objective_function(self, params: Dict[str, float]) -> float:
        """單目標函數：最大化干涉電路的整體性能"""
        design_params = DesignParameters(
            coupling_length=params['coupling_length'],
            gap=params['gap'],
            waveguide_width=params['waveguide_width']
        )
        
        try:
            result = self.simulator.simulate_classical(design_params)
            T = self.interferometer.compute_transmission_matrix(design_params)
            
            # 這裡我們簡化目標，主要關注保真度和均勻性
            # 理想矩陣（簡化）
            ideal_matrix = np.array([
                [1/np.sqrt(2), 1j/np.sqrt(2), 0],
                [1j/np.sqrt(2), 1/np.sqrt(2), 0],
                [0, 0, 1]
            ])
            fidelity = self.simulator._calculate_matrix_fidelity(T, ideal_matrix)
            
            # 簡易均勻性評估 (基於輸出功率)
            powers = [abs(T[i, 0])**2 + abs(T[i, 1])**2 for i in range(3)]
            uniformity = 1.0 - np.std(powers)

            composite_score = 0.7 * fidelity + 0.3 * uniformity
            
            self.optimization_history.append({
                'params': params.copy(),
                'fidelity': fidelity,
                'uniformity': uniformity,
                'composite_score': composite_score
            })
            
            return composite_score
        except Exception as e:
            print(f"模擬錯誤: {e}")
            return 0.0

    def run_optimization(self, n_iterations: int = 50) -> Dict:
        """執行設計最佳化"""
        print(f"=== 開始三輸入干涉電路設計最佳化 ===")
        print(f"迭代次數: {n_iterations}")

        bounds = {
            'coupling_length': (5.0, 30.0),
            'gap': (0.1, 0.8),
            'waveguide_width': (0.3, 0.7)
        }
        
        start_time = time.time()
        
        best_params, best_value, history = optimize_design(
            self.objective_function,
            bounds,
            n_iterations=n_iterations,
            verbose=True
        )
        
        optimization_time = time.time() - start_time
        
        self.best_result = {
            'params': best_params,
            'best_value': best_value,
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
        print(f"最佳綜合評分: {self.best_result['best_value']:.4f}")
        print(f"最佳化時間: {self.best_result['optimization_time']:.2f} 秒")
        
        print(f"\n最佳設計參數:")
        for param, value in self.best_result['params'].items():
            print(f"  {param}: {value:.4f}")

    def analyze_quantum_simulation(self):
        """對最佳參數執行並分析量子模擬"""
        if not self.best_result:
            print("尚未執行最佳化，無法進行量子模擬分析！")
            return
            
        print(f"\n=== 量子模擬分析 ===")
        print("使用找到的最佳參數，模擬雙光子輸入 |1,1,0> 的行為...")
        
        best_params_dict = self.best_result['params']
        best_design_params = DesignParameters(
            coupling_length=best_params_dict['coupling_length'],
            gap=best_params_dict['gap'],
            waveguide_width=best_params_dict['waveguide_width']
        )
        
        # 定義輸入態 |1,1,0>
        input_state_photons = [1, 1, 0]
        input_state_qobj = self.simulator.quantum_sim.create_fock_state(input_state_photons)

        # 執行量子模擬 (注意：此處的simulate_quantum是為2模態設計的，需擴展或簡化)
        # 我們將手動計算輸出態以展示原理
        T = self.interferometer.compute_transmission_matrix(best_design_params)
        
        # 對於 |1,1,0> 輸入，我們需要計算永久子來得到輸出概率
        # 這裡我們使用一個簡化的近似方法，直接從qutip的演化來計算
        # (一個更完整的實現需要一個多光子版本的simulate_quantum)
        
        # 創建玻色子算符
        a1 = self.simulator.quantum_sim.annihilation_operator(0)
        a2 = self.simulator.quantum_sim.annihilation_operator(1)
        a3 = self.simulator.quantum_sim.annihilation_operator(2)

        # 根據T構造演化算符U
        # 這是一個複雜的步驟，此處簡化為直接分析T
        # 實際的U是作用在多光子Fock空間上的，不是T本身
        
        print(f"\n⚛️ 量子模擬結果 (基於傳輸矩陣的理論分析):")
        print(f"  - 輸入態: |1,1,0> (第一和第二波導各一個光子)")

        # 計算輸出概率 (使用永久子)
        # 輸出 |2,0,0>: 兩個光子都在第一個端口
        U_sub_200 = T[[0,0], :][:, [0,1]]
        prob_200 = abs(U_sub_200[0,0]*U_sub_200[1,1] + U_sub_200[0,1]*U_sub_200[1,0])**2 / 2.0

        # 輸出 |0,2,0>: 兩個光子都在第二個端口
        U_sub_020 = T[[1,1], :][:, [0,1]]
        prob_020 = abs(U_sub_020[0,0]*U_sub_020[1,1] + U_sub_020[0,1]*U_sub_020[1,0])**2 / 2.0

        # 輸出 |1,1,0>: 兩個光子保持在原來的端口
        U_sub_110 = T[[0,1], :][:, [0,1]]
        prob_110 = abs(U_sub_110[0,0]*U_sub_110[1,1] + U_sub_110[0,1]*U_sub_110[1,0])**2

        # 輸出 |1,0,1>: 一個在1，一個在3
        U_sub_101 = T[[0,2], :][:, [0,1]]
        prob_101 = abs(U_sub_101[0,0]*U_sub_101[1,1] + U_sub_101[0,1]*U_sub_101[1,0])**2

        # 輸出 |0,1,1>: 一個在2，一個在3
        U_sub_011 = T[[1,2], :][:, [0,1]]
        prob_011 = abs(U_sub_011[0,0]*U_sub_011[1,1] + U_sub_011[0,1]*U_sub_011[1,0])**2

        # 歸一化
        total_prob = prob_200 + prob_020 + prob_110 + prob_101 + prob_011
        if total_prob > 0:
            prob_200 /= total_prob
            prob_020 /= total_prob
            prob_110 /= total_prob
            prob_101 /= total_prob
            prob_011 /= total_prob

        print(f"  - 輸出光子分佈概率:")
        print(f"    - |2,0,0>: {prob_200:.4f} (Hong-Ou-Mandel like bunching)")
        print(f"    - |0,2,0>: {prob_020:.4f} (Hong-Ou-Mandel like bunching)")
        print(f"    - |1,1,0>: {prob_110:.4f} (No interference)")
        print(f"    - |1,0,1>: {prob_101:.4f}")
        print(f"    - |0,1,1>: {prob_011:.4f}")

        # 繪圖
        labels = ['|2,0,0>', '|0,2,0>', '|1,1,0>', '|1,0,1>', '|0,1,1>']
        probs = [prob_200, prob_020, prob_110, prob_101, prob_011]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, probs, color='mediumpurple')
        ax.set_ylabel('概率')
        ax.set_title('最佳化干涉儀的雙光子輸出概率分佈 (輸入 |1,1,0>)')
        plt.show()

def main():
    """主函數"""
    print("="*60)
    print("案例B-量子版：三輸入干涉電路設計與量子模擬分析")
    print("="*60)
    
    # 1. 創建並執行最佳化
    optimizer = BosonSamplingOptimizer()
    optimizer.run_optimization(n_iterations=50)
    
    # 2. 分析經典模擬結果
    optimizer.analyze_classical_results()
    
    # 3. 執行並分析量子模擬
    optimizer.analyze_quantum_simulation()

if __name__ == "__main__":
    # 為了在qutip中正確處理多光子，需要擴展QuantumStateSimulator
    # 此處的實現是基於經典傳輸矩陣的理論計算，以展示物理概念
    # 在 `core/simulator.py` 中添加一個多光子演化函數將是下一步
    def Annihilation_Operator_Extention(self, mode_index: int):
        """創建指定模式的湮滅算符"""
        op_list = [qt.qeye(self.fock_dim)] * self.n_modes
        op_list[mode_index] = qt.destroy(self.fock_dim)
        return qt.tensor(op_list)
    
    from core.simulator import QuantumStateSimulator
    QuantumStateSimulator.annihilation_operator = Annihilation_Operator_Extention
    
    main()
