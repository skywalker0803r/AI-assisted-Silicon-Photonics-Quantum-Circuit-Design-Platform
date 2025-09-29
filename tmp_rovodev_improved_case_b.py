"""
改善版案例B：高品質三輸入干涉電路設計
Improved Case B: High-quality three-input interference circuit design
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

class ImprovedBosonSamplingOptimizer:
    """改善版Boson Sampling干涉電路最佳化器"""
    
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
    
    def compute_improved_ideal_matrix(self, params: DesignParameters) -> np.ndarray:
        """
        計算改善的理想矩陣，基於參數動態調整
        """
        # 使用Givens旋轉構建更realistic的理想矩陣
        theta1 = np.pi/4  # 45度混合
        theta2 = np.pi/3  # 60度混合
        phi = np.pi/2     # 90度相位
        
        # 構建更physical的3x3酉矩陣
        ideal_matrix = np.array([
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1)*np.cos(theta2), np.cos(theta1)*np.cos(theta2), -np.sin(theta2)],
            [np.sin(theta1)*np.sin(theta2)*np.exp(1j*phi), 
             np.cos(theta1)*np.sin(theta2)*np.exp(1j*phi), 
             np.cos(theta2)*np.exp(1j*phi)]
        ])
        
        return ideal_matrix
    
    def calculate_improved_fidelity(self, transmission_matrix: np.ndarray, 
                                  params: DesignParameters) -> float:
        """
        改善的保真度計算，考慮矩陣結構匹配
        """
        ideal_matrix = self.compute_improved_ideal_matrix(params)
        
        if transmission_matrix.shape != ideal_matrix.shape:
            return 0.0
        
        # 使用Frobenius norm計算矩陣相似度
        diff_matrix = transmission_matrix - ideal_matrix
        frobenius_error = np.linalg.norm(diff_matrix, 'fro')
        max_possible_error = np.linalg.norm(ideal_matrix, 'fro') * 2
        
        # 轉換為0-1範圍的保真度
        fidelity = max(0.0, 1.0 - frobenius_error / max_possible_error)
        
        # 額外檢查酉性
        unitarity_check = np.linalg.norm(
            transmission_matrix @ np.conj(transmission_matrix).T - np.eye(3), 'fro'
        )
        unitarity_penalty = min(unitarity_check, 1.0)
        
        final_fidelity = fidelity * (1.0 - 0.3 * unitarity_penalty)
        
        return float(max(0.0, final_fidelity))
    
    def calculate_improved_robustness(self, params: DesignParameters) -> float:
        """
        改善的製程容忍度計算
        """
        nominal_result = self.simulator.simulate_classical(params)
        nominal_T = self.interferometer.compute_transmission_matrix(params)
        
        # 測試多個擾動
        perturbations = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
        robustness_scores = []
        
        for pert in perturbations:
            # 對每個參數加入擾動
            perturbed_params = DesignParameters(
                coupling_length=params.coupling_length * (1 + pert),
                gap=params.gap * (1 + pert),
                waveguide_width=params.waveguide_width * (1 + pert),
                wavelength=params.wavelength
            )
            
            try:
                perturbed_result = self.simulator.simulate_classical(perturbed_params)
                perturbed_T = self.interferometer.compute_transmission_matrix(perturbed_params)
                
                # 計算性能變化
                transmission_change = abs(perturbed_result.transmission_efficiency - 
                                        nominal_result.transmission_efficiency)
                matrix_change = np.linalg.norm(perturbed_T - nominal_T, 'fro')
                
                # 計算容忍度分數
                transmission_tolerance = 1.0 / (1.0 + transmission_change / pert)
                matrix_tolerance = 1.0 / (1.0 + matrix_change / pert)
                
                robustness_scores.append((transmission_tolerance + matrix_tolerance) / 2)
                
            except:
                robustness_scores.append(0.1)  # 如果擾動導致失敗，給低分
        
        return float(np.mean(robustness_scores))
    
    def calculate_quantum_interference_quality(self, transmission_matrix: np.ndarray) -> float:
        """
        計算量子干涉品質指標
        """
        # 檢查矩陣的干涉特性
        # 1. 酉性檢查
        unitarity = np.linalg.norm(
            transmission_matrix @ np.conj(transmission_matrix).T - np.eye(3), 'fro'
        )
        unitarity_score = max(0.0, 1.0 - unitarity)
        
        # 2. 相位一致性
        phases = np.angle(transmission_matrix)
        phase_variance = np.var(phases[np.abs(transmission_matrix) > 0.1])
        phase_score = 1.0 / (1.0 + phase_variance)
        
        # 3. 幅度平衡
        amplitudes = np.abs(transmission_matrix)
        amplitude_uniformity = 1.0 - np.std(amplitudes.flatten())
        
        # 綜合評分
        quality = (0.5 * unitarity_score + 0.3 * phase_score + 0.2 * amplitude_uniformity)
        
        return float(max(0.0, min(1.0, quality)))
    
    def improved_single_objective_function(self, params: Dict[str, float]) -> float:
        """
        改善的單目標函數：更平衡的評分系統
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
            T = self.interferometer.compute_transmission_matrix(design_params)
            
            # 計算各項指標
            fidelity = self.calculate_improved_fidelity(T, design_params)
            robustness = self.calculate_improved_robustness(design_params)
            quantum_quality = self.calculate_quantum_interference_quality(T)
            transmission_eff = result.transmission_efficiency
            
            # 計算輸出概率分佈品質
            output_probs = self.calculate_output_probabilities(T)
            prob_values = list(output_probs.values())
            uniformity = float(1.0 - np.std(prob_values)) if prob_values else 0.0
            
            # 改善的綜合評分 - 更平衡的權重
            composite_score = (
                0.3 * fidelity +           # 降低保真度權重
                0.25 * quantum_quality +   # 新增量子干涉品質
                0.2 * robustness +         # 提高製程容忍度權重
                0.15 * uniformity +        # 輸出均勻性
                0.1 * transmission_eff     # 傳輸效率
            )
            
            # 記錄歷史
            self.optimization_history.append({
                'params': params.copy(),
                'fidelity': fidelity,
                'quantum_quality': quantum_quality,
                'robustness': robustness,
                'uniformity': uniformity,
                'transmission_eff': transmission_eff,
                'composite_score': composite_score,
                'output_probs': output_probs
            })
            
            return composite_score
            
        except Exception as e:
            print(f"模擬錯誤: {e}")
            return 0.0
    
    def calculate_output_probabilities(self, transmission_matrix: np.ndarray) -> Dict[str, float]:
        """計算輸出概率分佈"""
        def permanent_2x2(matrix):
            return matrix[0,0]*matrix[1,1] + matrix[0,1]*matrix[1,0]
        
        probabilities = {}
        
        # 計算各種輸出狀態的概率
        submatrix = transmission_matrix[:2, :2]
        prob_200 = abs(permanent_2x2(submatrix))**2
        probabilities['|2,0,0⟩'] = prob_200
        
        submatrix = transmission_matrix[[1,1], :][:, :2]
        prob_020 = abs(permanent_2x2(submatrix))**2
        probabilities['|0,2,0⟩'] = prob_020
        
        submatrix = transmission_matrix[:2, :2]
        prob_110 = abs(transmission_matrix[0,0] * transmission_matrix[1,1] - 
                      transmission_matrix[0,1] * transmission_matrix[1,0])**2
        probabilities['|1,1,0⟩'] = prob_110
        
        # 正規化
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    def run_improved_optimization(self, n_iterations: int = 150) -> Dict:
        """執行改善的最佳化"""
        print("=== 改善版單目標最佳化：高品質設計 ===")
        
        # 擴大搜索範圍
        bounds = {
            'coupling_length': (2.0, 50.0),   # 擴大範圍
            'gap': (0.05, 1.2),               # 擴大範圍
            'waveguide_width': (0.2, 1.0)     # 擴大範圍
        }
        
        start_time = time.time()
        
        best_params, best_value, history = optimize_design(
            self.improved_single_objective_function,
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
            'method': 'improved_single_objective'
        }
    
    def analyze_improved_results(self, result: Dict):
        """分析改善的結果"""
        print(f"\n=== 改善版結果分析 ===")
        
        print(f"最佳綜合評分: {result['best_value']:.4f} (目標: >0.7)")
        print(f"最佳化時間: {result['optimization_time']:.2f} 秒")
        print(f"最佳參數:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value:.4f}")
        
        # 分析最佳解的詳細特性
        if self.optimization_history:
            best_entry = max(self.optimization_history, key=lambda x: x['composite_score'])
            print(f"\n最佳解的詳細分析:")
            print(f"  保真度: {best_entry['fidelity']:.4f} (目標: >0.8)")
            print(f"  量子干涉品質: {best_entry['quantum_quality']:.4f} (目標: >0.8)")
            print(f"  製程容忍度: {best_entry['robustness']:.4f} (目標: >0.7)")
            print(f"  輸出均勻性: {best_entry['uniformity']:.4f} (目標: >0.8)")
            print(f"  傳輸效率: {best_entry['transmission_eff']:.4f} (目標: >0.95)")
            
            print(f"\n  品質評估:")
            total_targets_met = 0
            targets = [
                (best_entry['fidelity'], 0.8, "保真度"),
                (best_entry['quantum_quality'], 0.8, "量子干涉品質"),
                (best_entry['robustness'], 0.7, "製程容忍度"),
                (best_entry['uniformity'], 0.8, "輸出均勻性"),
                (best_entry['transmission_eff'], 0.95, "傳輸效率")
            ]
            
            for value, target, name in targets:
                status = "✅ 達標" if value >= target else "❌ 未達標"
                print(f"    {name}: {status}")
                if value >= target:
                    total_targets_met += 1
            
            print(f"\n  總體品質評級: {total_targets_met}/5 項指標達標")
            
            if total_targets_met >= 4:
                grade = "A級 - 優秀"
            elif total_targets_met >= 3:
                grade = "B級 - 良好"
            elif total_targets_met >= 2:
                grade = "C級 - 可接受"
            else:
                grade = "D級 - 需改善"
            
            print(f"  設計等級: {grade}")

def run_improved_case_b():
    """執行改善版案例B"""
    print("🚀 高品質三輸入干涉電路設計")
    print("=" * 60)
    
    # 創建改善版最佳化器
    optimizer = ImprovedBosonSamplingOptimizer()
    
    # 執行改善版最佳化
    print("\n🎯 執行改善版最佳化...")
    result = optimizer.run_improved_optimization(n_iterations=150)
    
    # 分析結果
    print("\n📊 分析結果...")
    optimizer.analyze_improved_results(result)
    
    return optimizer, result

if __name__ == "__main__":
    optimizer, result = run_improved_case_b()