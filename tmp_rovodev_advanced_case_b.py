"""
高級版案例B：多策略量子電路設計最佳化
Advanced Case B: Multi-strategy quantum circuit design optimization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from scipy.optimize import differential_evolution

from core.components import DesignParameters, ThreePortInterferometer
from core.simulator import CircuitSimulator
from optimization.bayesian_opt import optimize_design
from optimization.genetic_alg import GeneticOptimizer
from evaluation.metrics import MetricsCalculator, MultiObjectiveEvaluator
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

class AdvancedQuantumCircuitOptimizer:
    """高級量子電路最佳化器"""
    
    def __init__(self):
        self.setup_circuit()
        self.optimization_history = []
        
    def setup_circuit(self):
        """設置三端口干涉電路"""
        self.simulator = CircuitSimulator()
        self.interferometer = ThreePortInterferometer()
        self.simulator.add_component(self.interferometer)
        self.simulator.set_quantum_simulator(n_modes=3, n_photons=2)
    
    def create_adaptive_ideal_matrix(self, params: DesignParameters) -> np.ndarray:
        """
        創建自適應理想矩陣，根據實際參數調整期望
        """
        # 基於參數的自適應理想矩陣
        alpha = params.coupling_length / 25.0  # 正規化
        beta = params.gap / 0.6
        gamma = params.waveguide_width / 0.8
        
        # 使用參數化的Haar隨機酉矩陣方法
        theta1 = alpha * np.pi/2
        theta2 = beta * np.pi/3
        theta3 = gamma * np.pi/4
        
        phi1 = alpha * np.pi
        phi2 = beta * np.pi/2
        phi3 = gamma * np.pi/3
        
        # 構建參數化的3x3酉矩陣
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        c3, s3 = np.cos(theta3), np.sin(theta3)
        
        e1 = np.exp(1j * phi1)
        e2 = np.exp(1j * phi2)
        e3 = np.exp(1j * phi3)
        
        ideal_matrix = np.array([
            [c1*c2, -s1*c2*e1, s2*e2],
            [s1*c3, c1*c3*e1, s3*s2*e3],
            [-s1*s3, -c1*s3*e1, c3*c2*e3]
        ])
        
        # 確保矩陣是酉的
        U, S, Vh = np.linalg.svd(ideal_matrix)
        ideal_matrix = U @ Vh
        
        return ideal_matrix
    
    def compute_advanced_fidelity(self, transmission_matrix: np.ndarray, 
                                params: DesignParameters) -> float:
        """
        高級保真度計算，多重評估指標
        """
        # 1. 與自適應理想矩陣的保真度
        ideal_matrix = self.create_adaptive_ideal_matrix(params)
        fidelity1 = self.matrix_fidelity(transmission_matrix, ideal_matrix)
        
        # 2. 酉性保真度
        unitarity_error = np.linalg.norm(
            transmission_matrix @ np.conj(transmission_matrix).T - np.eye(3), 'fro'
        )
        fidelity2 = np.exp(-unitarity_error)
        
        # 3. 對稱性保真度（某些量子電路應具有對稱性）
        symmetry_error = np.linalg.norm(
            transmission_matrix - transmission_matrix.T.conj(), 'fro'
        )
        fidelity3 = np.exp(-symmetry_error / 2)
        
        # 4. 相位關聯性
        phases = np.angle(transmission_matrix.flatten())
        phase_coherence = 1.0 / (1.0 + np.std(phases))
        
        # 綜合保真度
        total_fidelity = (0.4 * fidelity1 + 0.3 * fidelity2 + 
                         0.2 * fidelity3 + 0.1 * phase_coherence)
        
        return float(max(0.0, min(1.0, total_fidelity)))
    
    def matrix_fidelity(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """計算兩個矩陣的保真度"""
        if matrix1.shape != matrix2.shape:
            return 0.0
        
        # 使用矩陣內積計算保真度
        inner_product = np.abs(np.trace(np.conj(matrix1).T @ matrix2))**2
        norm1 = np.trace(np.conj(matrix1).T @ matrix1)
        norm2 = np.trace(np.conj(matrix2).T @ matrix2)
        
        fidelity = inner_product / (norm1 * norm2)
        return float(np.real(fidelity))
    
    def compute_enhanced_robustness(self, params: DesignParameters) -> float:
        """
        增強的製程容忍度計算
        """
        try:
            nominal_T = self.interferometer.compute_transmission_matrix(params)
            nominal_result = self.simulator.simulate_classical(params)
            
            robustness_scores = []
            
            # 多種擾動測試
            perturbation_tests = [
                {'coupling_length': 0.02, 'gap': 0.0, 'waveguide_width': 0.0},
                {'coupling_length': 0.0, 'gap': 0.02, 'waveguide_width': 0.0},
                {'coupling_length': 0.0, 'gap': 0.0, 'waveguide_width': 0.02},
                {'coupling_length': 0.01, 'gap': 0.01, 'waveguide_width': 0.01},
                {'coupling_length': 0.03, 'gap': 0.03, 'waveguide_width': 0.03},
            ]
            
            for pert in perturbation_tests:
                perturbed_params = DesignParameters(
                    coupling_length=params.coupling_length * (1 + pert['coupling_length']),
                    gap=params.gap * (1 + pert['gap']),
                    waveguide_width=params.waveguide_width * (1 + pert['waveguide_width']),
                    wavelength=params.wavelength
                )
                
                try:
                    perturbed_T = self.interferometer.compute_transmission_matrix(perturbed_params)
                    perturbed_result = self.simulator.simulate_classical(perturbed_params)
                    
                    # 多重容忍度指標
                    matrix_stability = 1.0 / (1.0 + np.linalg.norm(perturbed_T - nominal_T, 'fro'))
                    
                    transmission_stability = 1.0 / (1.0 + abs(
                        perturbed_result.transmission_efficiency - 
                        nominal_result.transmission_efficiency
                    ))
                    
                    fidelity_stability = self.matrix_fidelity(perturbed_T, nominal_T)
                    
                    combined_stability = (matrix_stability + transmission_stability + fidelity_stability) / 3
                    robustness_scores.append(combined_stability)
                    
                except:
                    robustness_scores.append(0.1)
            
            return float(np.mean(robustness_scores))
            
        except Exception as e:
            return 0.1
    
    def compute_quantum_advantage_metric(self, transmission_matrix: np.ndarray) -> float:
        """
        計算量子優勢指標
        """
        # 1. 非經典關聯性
        eigenvals = np.linalg.eigvals(transmission_matrix @ np.conj(transmission_matrix).T)
        eigenval_entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-10))
        nonclassical_score = min(eigenval_entropy / np.log(3), 1.0)
        
        # 2. 干涉可見度
        amplitudes = np.abs(transmission_matrix)
        visibility = (np.max(amplitudes) - np.min(amplitudes)) / (np.max(amplitudes) + np.min(amplitudes))
        
        # 3. 相位分散
        phases = np.angle(transmission_matrix)
        phase_dispersion = np.std(phases)
        phase_score = min(phase_dispersion / np.pi, 1.0)
        
        quantum_advantage = (0.5 * nonclassical_score + 0.3 * visibility + 0.2 * phase_score)
        
        return float(max(0.0, min(1.0, quantum_advantage)))
    
    def advanced_objective_function(self, params_array: np.ndarray) -> float:
        """
        高級目標函數，用於differential evolution
        """
        params_dict = {
            'coupling_length': params_array[0],
            'gap': params_array[1], 
            'waveguide_width': params_array[2]
        }
        
        design_params = DesignParameters(
            coupling_length=params_dict['coupling_length'],
            gap=params_dict['gap'],
            waveguide_width=params_dict['waveguide_width'],
            wavelength=1550e-9
        )
        
        try:
            # 計算傳輸矩陣
            T = self.interferometer.compute_transmission_matrix(design_params)
            result = self.simulator.simulate_classical(design_params)
            
            # 高級指標計算
            fidelity = self.compute_advanced_fidelity(T, design_params)
            robustness = self.compute_enhanced_robustness(design_params)
            quantum_advantage = self.compute_quantum_advantage_metric(T)
            
            # 輸出概率品質
            output_probs = self.calculate_output_probabilities(T)
            prob_values = list(output_probs.values())
            uniformity = float(1.0 - np.std(prob_values)) if prob_values else 0.0
            
            # 傳輸效率
            transmission_eff = result.transmission_efficiency
            
            # 新的綜合評分策略
            composite_score = (
                0.35 * fidelity +          # 主要：保真度
                0.25 * robustness +        # 重要：製程容忍度
                0.2 * quantum_advantage +  # 新增：量子優勢
                0.15 * uniformity +        # 輸出品質
                0.05 * transmission_eff    # 基礎：傳輸效率
            )
            
            # 記錄歷史（轉換為最小化問題）
            self.optimization_history.append({
                'params': params_dict.copy(),
                'fidelity': fidelity,
                'robustness': robustness,
                'quantum_advantage': quantum_advantage,
                'uniformity': uniformity,
                'transmission_eff': transmission_eff,
                'composite_score': composite_score,
                'output_probs': output_probs
            })
            
            # 返回負值（因為differential evolution是最小化）
            return -composite_score
            
        except Exception as e:
            return 0.0  # 失敗返回最差值
    
    def calculate_output_probabilities(self, transmission_matrix: np.ndarray) -> Dict[str, float]:
        """計算輸出概率分佈"""
        def permanent_2x2(matrix):
            return matrix[0,0]*matrix[1,1] + matrix[0,1]*matrix[1,0]
        
        probabilities = {}
        
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
        
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v/total_prob for k, v in probabilities.items()}
        
        return probabilities
    
    def run_advanced_optimization(self) -> Dict:
        """執行高級最佳化"""
        print("=== 高級多策略最佳化：追求A級設計 ===")
        
        # 定義搜索邊界
        bounds = [
            (1.0, 100.0),    # coupling_length - 大幅擴大
            (0.01, 2.0),     # gap - 大幅擴大  
            (0.1, 2.0)       # waveguide_width - 大幅擴大
        ]
        
        start_time = time.time()
        
        # 使用differential evolution進行全域最佳化
        result = differential_evolution(
            self.advanced_objective_function,
            bounds,
            maxiter=200,
            popsize=20,
            seed=42,
            disp=True,
            polish=True,
            atol=1e-8,
            tol=1e-8
        )
        
        optimization_time = time.time() - start_time
        
        best_params = {
            'coupling_length': result.x[0],
            'gap': result.x[1],
            'waveguide_width': result.x[2]
        }
        
        best_value = -result.fun  # 轉回最大化問題
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'optimization_time': optimization_time,
            'method': 'advanced_differential_evolution',
            'success': result.success,
            'n_iterations': result.nit
        }
    
    def analyze_advanced_results(self, result: Dict):
        """分析高級結果"""
        print(f"\n=== 高級最佳化結果分析 ===")
        
        print(f"✅ 最佳化成功: {'是' if result.get('success', False) else '否'}")
        print(f"🎯 最佳綜合評分: {result['best_value']:.4f} (目標: >0.8)")
        print(f"⏱️ 最佳化時間: {result['optimization_time']:.2f} 秒")
        print(f"🔄 迭代次數: {result.get('n_iterations', 'N/A')}")
        
        print(f"\n📐 最佳參數:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value:.4f}")
        
        # 分析最佳解的詳細特性
        if self.optimization_history:
            best_entry = max(self.optimization_history, key=lambda x: x['composite_score'])
            print(f"\n🔬 最佳解的詳細分析:")
            print(f"  🎯 保真度: {best_entry['fidelity']:.4f} (目標: >0.8)")
            print(f"  🛡️ 製程容忍度: {best_entry['robustness']:.4f} (目標: >0.8)")
            print(f"  ⚛️ 量子優勢: {best_entry['quantum_advantage']:.4f} (目標: >0.7)")
            print(f"  ⚖️ 輸出均勻性: {best_entry['uniformity']:.4f} (目標: >0.8)")
            print(f"  📡 傳輸效率: {best_entry['transmission_eff']:.4f} (目標: >0.95)")
            
            print(f"\n📊 品質評估:")
            targets = [
                (best_entry['fidelity'], 0.8, "保真度"),
                (best_entry['robustness'], 0.8, "製程容忍度"), 
                (best_entry['quantum_advantage'], 0.7, "量子優勢"),
                (best_entry['uniformity'], 0.8, "輸出均勻性"),
                (best_entry['transmission_eff'], 0.95, "傳輸效率")
            ]
            
            total_targets_met = 0
            for value, target, name in targets:
                status = "✅ 達標" if value >= target else "❌ 未達標"
                percentage = f"({value/target*100:.1f}%)"
                print(f"    {name}: {status} {percentage}")
                if value >= target:
                    total_targets_met += 1
            
            print(f"\n🏆 總體品質評級: {total_targets_met}/5 項指標達標")
            
            if total_targets_met >= 4:
                grade = "A級 - 優秀 🥇"
                recommendation = "設計已達到產業級標準，可進入製造階段"
            elif total_targets_met >= 3:
                grade = "B級 - 良好 🥈"
                recommendation = "設計品質良好，建議小幅調整後製造"
            elif total_targets_met >= 2:
                grade = "C級 - 可接受 🥉"
                recommendation = "基本可用，建議進一步最佳化"
            else:
                grade = "D級 - 需改善 📝"
                recommendation = "需要重新設計或調整策略"
            
            print(f"  🎖️ 設計等級: {grade}")
            print(f"  💡 建議: {recommendation}")
            
            # 輸出概率分析
            print(f"\n🔀 量子態輸出概率分佈:")
            for state, prob in best_entry['output_probs'].items():
                print(f"    {state}: {prob:.4f}")

def run_advanced_case_b():
    """執行高級版案例B"""
    print("🚀 高級量子電路設計最佳化")
    print("=" * 80)
    print("目標：設計出A級品質的量子干涉電路")
    print("=" * 80)
    
    # 創建高級最佳化器
    optimizer = AdvancedQuantumCircuitOptimizer()
    
    # 執行高級最佳化
    print("\n🔥 執行高級多策略最佳化...")
    result = optimizer.run_advanced_optimization()
    
    # 分析結果
    print("\n📈 分析最終結果...")
    optimizer.analyze_advanced_results(result)
    
    return optimizer, result

if __name__ == "__main__":
    optimizer, result = run_advanced_case_b()