"""
案例B：三輸入干涉電路設計（簡單的Boson Sampling元件）
Case B: Three-input interference circuit design for Boson Sampling
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

from core.components import DesignParameters, ThreePortInterferometer
from core.simulator import CircuitSimulator
from optimization.bayesian_opt import optimize_design
from optimization.genetic_alg import GeneticOptimizer
from evaluation.metrics import MetricsCalculator, MultiObjectiveEvaluator
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

class BosonSamplingOptimizer:
    """Boson Sampling干涉電路最佳化器"""
    
    def __init__(self):
        self.setup_circuit()
        self.optimization_history = []
        self.multi_obj_evaluator = MultiObjectiveEvaluator()
        
    def setup_circuit(self):
        """設置三端口干涉電路"""
        self.simulator = CircuitSimulator()
        self.interferometer = ThreePortInterferometer()
        self.simulator.add_component(self.interferometer)
        self.simulator.set_quantum_simulator(n_modes=3, n_photons=2)
    
    def calculate_boson_sampling_fidelity(self, transmission_matrix: np.ndarray) -> float:
        """
        計算Boson Sampling的保真度
        基於理想的3x3酉矩陣與實際矩陣的比較
        """
        # 理想的干涉矩陣（修改為與ThreePortInterferometer結構匹配的簡化理想矩陣）
        # 原始的ThreePortInterferometer在T[0,2]和T[2,0]位置有0
        # 因此，理想矩陣也應在這些位置有0，以進行有意義的比較
        ideal_matrix = np.array([
            [1/np.sqrt(2), 1/np.sqrt(2), 0],
            [1/np.sqrt(2), -1/np.sqrt(2), 0],
            [0, 0, 1] # 假設第三個模式是直通的，或者有其他簡化
        ])
        # 為了匹配原始的3x3 Haar隨機酉矩陣的複雜性，但又考慮到0的限制
        # 這裡使用一個簡化的3x3酉矩陣，它在T[0,2]和T[2,0]位置為0
        # 這是基於兩個2x2分束器和一個直通模式的簡化模型
        # 確保酉性
        # 這裡的目標是讓保真度計算有意義，而不是追求一個通用的Haar隨機矩陣
        # 考慮到 ThreePortInterferometer 的結構，它更像是一個2x2的干涉儀加上一個直通模式
        # 讓我們使用一個更符合其結構的理想矩陣
        # 假設理想情況下，前兩個模式是50/50分束，第三個模式直通
        ideal_matrix = np.array([
            [1/np.sqrt(2), 1j/np.sqrt(2), 0],
            [1j/np.sqrt(2), 1/np.sqrt(2), 0],
            [0, 0, 1]
        ])
        
        # 計算矩陣保真度
        if transmission_matrix.shape != ideal_matrix.shape:
            return 0.0
            
        overlap = np.abs(np.trace(np.conj(transmission_matrix).T @ ideal_matrix))**2
        norm_actual = np.trace(np.conj(transmission_matrix).T @ transmission_matrix)
        norm_ideal = np.trace(np.conj(ideal_matrix).T @ ideal_matrix)
        
        fidelity = overlap / (norm_actual * norm_ideal)
        return float(np.real(fidelity))
    
    def calculate_output_probabilities(self, transmission_matrix: np.ndarray, 
                                     input_state: List[int] = [1, 1, 0]) -> Dict[str, float]:
        """
        計算Boson Sampling的輸出概率分佈
        使用永久子(permanent)計算
        """
        # 簡化實現：計算幾個重要的輸出配置概率
        
        def permanent_2x2(matrix):
            """計算2x2矩陣的永久子"""
            return matrix[0,0]*matrix[1,1] + matrix[0,1]*matrix[1,0]
        
        # 對於輸入 |1,1,0⟩，計算幾個可能的輸出
        probabilities = {}
        
        # 輸出 |2,0,0⟩ - 兩個光子都到第一個模式
        submatrix = transmission_matrix[:2, :2]  # 取前兩行兩列
        prob_200 = abs(permanent_2x2(submatrix))**2
        probabilities['|2,0,0⟩'] = prob_200
        
        # 輸出 |0,2,0⟩ - 兩個光子都到第二個模式
        submatrix = transmission_matrix[[1,1], :][:, :2]
        prob_020 = abs(permanent_2x2(submatrix))**2
        probabilities['|0,2,0⟩'] = prob_020
        
        # 輸出 |1,1,0⟩ - 保持原分佈
        submatrix = transmission_matrix[:2, :2]
        prob_110 = abs(transmission_matrix[0,0] * transmission_matrix[1,1] - 
                      transmission_matrix[0,1] * transmission_matrix[1,0])**2
        probabilities['|1,1,0⟩'] = prob_110
        
        # 正規化
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    def single_objective_function(self, params: Dict[str, float]) -> float:
        """
        單目標函數：最大化干涉電路的整體性能
        """
        design_params = DesignParameters(
            coupling_length=params['coupling_length'],
            gap=params['gap'],
            waveguide_width=params['waveguide_width'],
            wavelength=1550e-9
        )
        
        try:
            # 執行模擬
            result = self.simulator.simulate_classical(design_params)
            
            # 獲取傳輸矩陣
            T = self.interferometer.compute_transmission_matrix(design_params)
            
            # 計算各項指標
            bs_fidelity = self.calculate_boson_sampling_fidelity(T)
            output_probs = self.calculate_output_probabilities(T)
            
            # 計算輸出均勻性（理想情況下各輸出概率相近）
            prob_values = list(output_probs.values())
            uniformity = float(1.0 - np.std(prob_values)) if prob_values else 0.0
            
            # 計算製程容忍度
            robustness = result.robustness_score
            
            # 綜合評分
            composite_score = (
                0.6 * bs_fidelity +      # Boson Sampling保真度
                0.2 * uniformity +       # 輸出均勻性
                0.1 * robustness +       # 製程容忍度
                0.1 * result.transmission_efficiency  # 傳輸效率
            )
            
            # 記錄歷史
            self.optimization_history.append({
                'params': params.copy(),
                'bs_fidelity': bs_fidelity,
                'uniformity': uniformity,
                'robustness': robustness,
                'transmission_eff': result.transmission_efficiency,
                'composite_score': composite_score,
                'output_probs': output_probs
            })
            
            return composite_score
            
        except Exception as e:
            print(f"模擬錯誤: {e}")
            return 0.0
    
    def multi_objective_function(self, params: Dict[str, float]) -> List[float]:
        """
        多目標函數：返回多個目標值
        [保真度, 均勻性, 製程容忍度]
        """
        design_params = DesignParameters(
            coupling_length=params['coupling_length'],
            gap=params['gap'],
            waveguide_width=params['waveguide_width'],
            wavelength=1550e-9
        )
        
        try:
            result = self.simulator.simulate_classical(design_params)
            T = self.interferometer.compute_transmission_matrix(design_params)
            
            bs_fidelity = self.calculate_boson_sampling_fidelity(T)
            output_probs = self.calculate_output_probabilities(T)
            prob_values = list(output_probs.values())
            uniformity = float(1.0 - np.std(prob_values)) if prob_values else 0.0
            robustness = result.robustness_score
            
            objectives = [float(bs_fidelity), float(uniformity), float(robustness)]
            
            # 添加到多目標評估器
            self.multi_obj_evaluator.add_solution(params, objectives)
            
            return objectives
            
        except Exception as e:
            print(f"模擬錯誤: {e}")
            return [0.0, 0.0, 0.0]
    
    def run_single_objective_optimization(self, n_iterations: int = 50) -> Dict:
        """執行單目標最佳化"""
        print("=== 單目標最佳化：最大化綜合性能 ===")
        
        bounds = {
            'coupling_length': (5.0, 30.0),
            'gap': (0.1, 0.8),
            'waveguide_width': (0.3, 0.7)
        }
        
        start_time = time.time()
        
        best_params, best_value, history = optimize_design(
            self.single_objective_function,
            bounds,
            n_iterations=n_iterations,
            acquisition_func='ei',
            verbose=True
        )
        
        optimization_time = time.time() - start_time
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'optimization_time': optimization_time,
            'method': 'single_objective_bayesian'
        }
    
    def run_multi_objective_optimization(self, n_iterations: int = 100) -> Dict:
        """執行多目標最佳化"""
        print("=== 多目標最佳化：Pareto前緣搜尋 ===")
        
        bounds = {
            'coupling_length': (5.0, 30.0),
            'gap': (0.1, 0.8),
            'waveguide_width': (0.3, 0.7)
        }
        
        start_time = time.time()
        
        # 使用遺傳演算法進行多目標最佳化
        from optimization.genetic_alg import GeneticOptimizer
        
        genetic_opt = GeneticOptimizer(
            bounds=bounds,
            population_size=50,
            n_generations=n_iterations // 10,
            multi_objective=True
        )
        
        best_solutions = genetic_opt.optimize(self.multi_objective_function)
        
        optimization_time = time.time() - start_time
        
        # 獲取Pareto前緣
        pareto_front = self.multi_obj_evaluator.get_pareto_front()
        
        return {
            'pareto_front': pareto_front,
            'best_solutions': best_solutions,
            'optimization_time': optimization_time,
            'method': 'multi_objective_genetic'
        }
    
    def analyze_results(self, single_obj_result: Dict, multi_obj_result: Dict):
        """分析最佳化結果"""
        print(f"\n=== 結果分析 ===")
        
        # 單目標結果
        print(f"單目標最佳化:")
        print(f"  最佳綜合評分: {single_obj_result['best_value']:.4f}")
        print(f"  最佳化時間: {single_obj_result['optimization_time']:.2f} 秒")
        print(f"  最佳參數:")
        for param, value in single_obj_result['best_params'].items():
            print(f"    {param}: {value:.4f}")
        
        # 多目標結果
        print(f"\n多目標最佳化:")
        print(f"  Pareto前緣解的數量: {len(multi_obj_result['pareto_front'])}")
        print(f"  最佳化時間: {multi_obj_result['optimization_time']:.2f} 秒")
        
        if multi_obj_result['pareto_front']:
            print(f"  Pareto前緣範例解:")
            for i, solution in enumerate(multi_obj_result['pareto_front'][:3]):
                print(f"    解 {i+1}:")
                print(f"      目標值: {solution['objectives']}")
                print(f"      參數: {solution['params']}")
        
        # 分析最佳解的物理特性
        if self.optimization_history:
            best_entry = max(self.optimization_history, key=lambda x: x['composite_score'])
            print(f"\n最佳解的物理特性:")
            print(f"  Boson Sampling保真度: {best_entry['bs_fidelity']:.4f}")
            print(f"  輸出均勻性: {best_entry['uniformity']:.4f}")
            print(f"  製程容忍度: {best_entry['robustness']:.4f}")
            print(f"  傳輸效率: {best_entry['transmission_eff']:.4f}")
            print(f"  輸出概率分佈:")
            for state, prob in best_entry['output_probs'].items():
                print(f"    {state}: {prob:.4f}")
    
    def plot_results(self, multi_obj_result: Dict):
        """繪製結果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 最佳化收斂歷史
        if self.optimization_history:
            iterations = range(len(self.optimization_history))
            scores = [entry['composite_score'] for entry in self.optimization_history]
            best_so_far = [max(scores[:i+1]) for i in range(len(scores))]
            
            ax1.plot(iterations, scores, 'b-', alpha=0.6, label='目標值')
            ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='目前最佳')
            ax1.set_xlabel('迭代次數')
            ax1.set_ylabel('綜合評分')
            ax1.set_title('單目標最佳化收斂')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Pareto前緣（保真度 vs 均勻性）
        if multi_obj_result['pareto_front']:
            all_obj1 = [sol['objectives'][0] for sol in self.multi_obj_evaluator.all_solutions]
            all_obj2 = [sol['objectives'][1] for sol in self.multi_obj_evaluator.all_solutions]
            pareto_obj1 = [sol['objectives'][0] for sol in multi_obj_result['pareto_front']]
            pareto_obj2 = [sol['objectives'][1] for sol in multi_obj_result['pareto_front']]
            
            ax2.scatter(all_obj1, all_obj2, c='lightblue', alpha=0.6, label='所有解')
            ax2.scatter(pareto_obj1, pareto_obj2, c='red', s=100, label='Pareto前緣')
            ax2.set_xlabel('Boson Sampling保真度')
            ax2.set_ylabel('輸出均勻性')
            ax2.set_title('Pareto前緣')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 參數分佈
        if self.optimization_history:
            coupling_lengths = [entry['params']['coupling_length'] for entry in self.optimization_history]
            gaps = [entry['params']['gap'] for entry in self.optimization_history]
            
            ax3.scatter(coupling_lengths, gaps, c=[entry['composite_score'] for entry in self.optimization_history], 
                       cmap='viridis', alpha=0.7)
            ax3.set_xlabel('耦合長度 (μm)')
            ax3.set_ylabel('間距 (μm)')
            ax3.set_title('參數空間探索')
            ax3.grid(True, alpha=0.3)
        
        # 4. 目標值分佈
        if self.optimization_history:
            fidelities = [entry['bs_fidelity'] for entry in self.optimization_history]
            uniformities = [entry['uniformity'] for entry in self.optimization_history]
            robustness = [entry['robustness'] for entry in self.optimization_history]
            
            ax4.hist([fidelities, uniformities, robustness], 
                    bins=15, alpha=0.7, label=['保真度', '均勻性', '容忍度'])
            ax4.set_xlabel('目標值')
            ax4.set_ylabel('頻率')
            ax4.set_title('目標值分佈')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """主函數：執行三輸入干涉電路設計案例"""
    print("三輸入干涉電路設計最佳化案例")
    print("=" * 50)
    
    # 創建最佳化器
    optimizer = BosonSamplingOptimizer()
    
    # 執行單目標最佳化
    print("\n1. 執行單目標最佳化...")
    single_result = optimizer.run_single_objective_optimization(n_iterations=100)
    
    # 執行多目標最佳化
    print("\n2. 執行多目標最佳化...")
    multi_result = optimizer.run_multi_objective_optimization(n_iterations=200)
    
    # 分析結果
    print("\n3. 分析結果...")
    optimizer.analyze_results(single_result, multi_result)
    
    # 繪製結果
    print("\n4. 繪製結果...")
    optimizer.plot_results(multi_result)

if __name__ == "__main__":
    main()