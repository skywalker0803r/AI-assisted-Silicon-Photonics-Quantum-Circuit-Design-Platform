"""
Performance metrics for silicon photonics design evaluation
性能指標評估模組
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """性能指標數據結構"""
    transmission_efficiency: float
    splitting_ratio_error: float
    insertion_loss_db: float
    fidelity: float
    robustness_score: float
    bandwidth_nm: float
    
class MetricsCalculator:
    """指標計算器"""
    
    def __init__(self):
        self.target_specs = {}
    
    def set_target_specs(self, **specs):
        """設置目標規格"""
        self.target_specs.update(specs)
    
    def calculate_transmission_efficiency(self, power_in: float, power_out: float) -> float:
        """計算傳輸效率"""
        if power_in <= 0:
            return 0.0
        return power_out / power_in
    
    def calculate_splitting_ratio_error(self, actual_ratios: List[float], 
                                      target_ratios: List[float]) -> float:
        """計算分束比誤差"""
        if len(actual_ratios) != len(target_ratios):
            raise ValueError("Ratio arrays must have same length")
        
        errors = []
        for actual, target in zip(actual_ratios, target_ratios):
            if target > 0:
                errors.append(abs(actual - target) / target)
            else:
                errors.append(abs(actual - target))
        
        return np.mean(errors)
    
    def calculate_insertion_loss(self, power_in: float, power_out: float) -> float:
        """計算插入損耗 (dB)"""
        if power_out <= 0 or power_in <= 0:
            return float('inf')
        return -10 * np.log10(power_out / power_in)
    
    def calculate_fidelity(self, actual_matrix: np.ndarray, 
                          ideal_matrix: np.ndarray) -> float:
        """計算矩陣保真度"""
        if actual_matrix.shape != ideal_matrix.shape:
            return 0.0
        
        # 歸一化矩陣
        actual_norm = actual_matrix / np.linalg.norm(actual_matrix, 'fro')
        ideal_norm = ideal_matrix / np.linalg.norm(ideal_matrix, 'fro')
        
        # 計算重疊度
        overlap = np.abs(np.trace(np.conj(actual_norm).T @ ideal_norm))**2
        normalization = np.trace(np.conj(actual_norm).T @ actual_norm) * \
                       np.trace(np.conj(ideal_norm).T @ ideal_norm)
        
        return float(np.real(overlap / normalization))
    
    def calculate_robustness_score(self, nominal_result, perturbed_results: List,
                                 perturbation_levels: List[float]) -> float:
        """計算製程容忍度分數"""
        if len(perturbed_results) != len(perturbation_levels):
            raise ValueError("Results and perturbation levels must match")
        
        robustness_scores = []
        
        for result, level in zip(perturbed_results, perturbation_levels):
            # 計算性能變化
            if hasattr(nominal_result, 'transmission_efficiency'):
                nominal_eff = nominal_result.transmission_efficiency
                perturbed_eff = result.transmission_efficiency
                
                relative_change = abs(perturbed_eff - nominal_eff) / (nominal_eff + 1e-9)
                sensitivity = relative_change / level  # 敏感度
                
                # 容忍度分數 (越小越好)
                robustness_scores.append(1.0 / (1.0 + sensitivity))
            else:
                robustness_scores.append(0.5)  # 預設中等分數
        
        return float(np.mean(robustness_scores))
    
    def calculate_bandwidth(self, wavelengths: np.ndarray, 
                          transmission: np.ndarray, 
                          threshold_db: float = 3.0) -> float:
        """計算3dB頻寬"""
        if len(wavelengths) != len(transmission):
            raise ValueError("Wavelength and transmission arrays must match")
        
        # 轉換為dB
        transmission_db = 10 * np.log10(transmission + 1e-9)
        max_transmission_db = np.max(transmission_db)
        
        # 找到3dB點
        threshold = max_transmission_db - threshold_db
        above_threshold = transmission_db >= threshold
        
        if not np.any(above_threshold):
            return 0.0
        
        # 找到頻寬範圍
        indices = np.where(above_threshold)[0]
        bandwidth = wavelengths[indices[-1]] - wavelengths[indices[0]]
        
        return float(bandwidth * 1e9)  # 轉換為nm
    
    def evaluate_design(self, simulation_result, target_specs: Optional[Dict] = None) -> PerformanceMetrics:
        """評估設計性能"""
        if target_specs:
            self.set_target_specs(**target_specs)
        
        # 提取基本指標
        transmission_eff = simulation_result.transmission_efficiency
        loss_db = simulation_result.loss_db
        fidelity = simulation_result.fidelity
        robustness = simulation_result.robustness_score
        
        # 計算分束比誤差
        if hasattr(simulation_result, 'splitting_ratio') and 'target_splitting_ratio' in self.target_specs:
            target_ratios = self.target_specs['target_splitting_ratio']
            actual_ratios = list(simulation_result.splitting_ratio)
            splitting_error = self.calculate_splitting_ratio_error(actual_ratios, target_ratios)
        else:
            splitting_error = 0.0
        
        # 計算頻寬
        if hasattr(simulation_result, 'wavelength_response'):
            wavelengths = np.linspace(1500e-9, 1600e-9, len(simulation_result.wavelength_response))
            bandwidth = self.calculate_bandwidth(wavelengths, simulation_result.wavelength_response)
        else:
            bandwidth = 0.0
        
        return PerformanceMetrics(
            transmission_efficiency=transmission_eff,
            splitting_ratio_error=splitting_error,
            insertion_loss_db=loss_db,
            fidelity=fidelity,
            robustness_score=robustness,
            bandwidth_nm=bandwidth
        )
    
    def calculate_composite_score(self, metrics: PerformanceMetrics, 
                                weights: Optional[Dict[str, float]] = None) -> float:
        """計算綜合評分"""
        if weights is None:
            weights = {
                'transmission_efficiency': 0.3,
                'splitting_ratio_error': 0.2,
                'insertion_loss_db': 0.2,
                'fidelity': 0.15,
                'robustness_score': 0.1,
                'bandwidth_nm': 0.05
            }
        
        # 正規化各項指標 (0-1範圍)
        eff_score = min(metrics.transmission_efficiency, 1.0)
        
        # 分束比誤差 (越小越好)
        ratio_score = max(0.0, 1.0 - metrics.splitting_ratio_error)
        
        # 插入損耗 (越小越好，假設目標<1dB)
        loss_score = max(0.0, 1.0 - metrics.insertion_loss_db / 5.0)
        
        # 保真度
        fidelity_score = metrics.fidelity
        
        # 製程容忍度
        robustness_score = metrics.robustness_score
        
        # 頻寬 (正規化到50nm)
        bandwidth_score = min(metrics.bandwidth_nm / 50.0, 1.0)
        
        # 加權總分
        total_score = (
            weights['transmission_efficiency'] * eff_score +
            weights['splitting_ratio_error'] * ratio_score +
            weights['insertion_loss_db'] * loss_score +
            weights['fidelity'] * fidelity_score +
            weights['robustness_score'] * robustness_score +
            weights['bandwidth_nm'] * bandwidth_score
        )
        
        return float(total_score)

class MultiObjectiveEvaluator:
    """多目標評估器"""
    
    def __init__(self):
        self.pareto_front = []
        self.all_solutions = []
    
    def add_solution(self, params: Dict, objectives: List[float]):
        """添加解到評估中"""
        solution = {
            'params': params.copy(),
            'objectives': objectives.copy(),
            'is_pareto': False
        }
        self.all_solutions.append(solution)
        self._update_pareto_front()
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """檢查obj1是否支配obj2 (假設最大化所有目標)"""
        better_in_any = False
        for o1, o2 in zip(obj1, obj2):
            if o1 < o2:
                return False
            elif o1 > o2:
                better_in_any = True
        return better_in_any
    
    def _update_pareto_front(self):
        """更新Pareto前緣"""
        # 重置所有解的Pareto狀態
        for sol in self.all_solutions:
            sol['is_pareto'] = True
        
        # 檢查支配關係
        for i, sol1 in enumerate(self.all_solutions):
            for j, sol2 in enumerate(self.all_solutions):
                if i != j and self._dominates(sol2['objectives'], sol1['objectives']):
                    sol1['is_pareto'] = False
                    break
        
        # 更新Pareto前緣
        self.pareto_front = [sol for sol in self.all_solutions if sol['is_pareto']]
    
    def get_pareto_front(self) -> List[Dict]:
        """獲取Pareto前緣"""
        return self.pareto_front.copy()
    
    def plot_pareto_front(self, obj_names: List[str] = None):
        """繪製Pareto前緣 (僅支援2D)"""
        if len(self.all_solutions) == 0:
            print("No solutions to plot")
            return
        
        if len(self.all_solutions[0]['objectives']) != 2:
            print("Pareto front plotting only supports 2 objectives")
            return
        
        # 提取目標值
        all_obj1 = [sol['objectives'][0] for sol in self.all_solutions]
        all_obj2 = [sol['objectives'][1] for sol in self.all_solutions]
        
        pareto_obj1 = [sol['objectives'][0] for sol in self.pareto_front]
        pareto_obj2 = [sol['objectives'][1] for sol in self.pareto_front]
        
        # 繪圖
        plt.figure(figsize=(10, 8))
        plt.scatter(all_obj1, all_obj2, c='lightblue', alpha=0.6, label='All Solutions')
        plt.scatter(pareto_obj1, pareto_obj2, c='red', s=100, label='Pareto Front')
        
        # 連接Pareto前緣點
        if len(pareto_front) > 1:
            sorted_pareto = sorted(zip(pareto_obj1, pareto_obj2))
            pareto_x, pareto_y = zip(*sorted_pareto)
            plt.plot(pareto_x, pareto_y, 'r--', alpha=0.7)
        
        plt.xlabel(obj_names[0] if obj_names else 'Objective 1')
        plt.ylabel(obj_names[1] if obj_names else 'Objective 2')
        plt.title('Pareto Front')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# 便利函數
def quick_evaluate(simulation_result, target_splitting_ratio: List[float] = [0.5, 0.5]) -> Dict:
    """快速評估函數"""
    calculator = MetricsCalculator()
    calculator.set_target_specs(target_splitting_ratio=target_splitting_ratio)
    
    metrics = calculator.evaluate_design(simulation_result)
    composite_score = calculator.calculate_composite_score(metrics)
    
    return {
        'metrics': metrics,
        'composite_score': composite_score,
        'summary': {
            'transmission': f"{metrics.transmission_efficiency:.3f}",
            'loss_db': f"{metrics.insertion_loss_db:.2f}",
            'fidelity': f"{metrics.fidelity:.3f}",
            'robustness': f"{metrics.robustness_score:.3f}",
            'score': f"{composite_score:.3f}"
        }
    }

# 測試範例
if __name__ == "__main__":
    # 模擬一些測試數據
    class MockResult:
        def __init__(self):
            self.transmission_efficiency = 0.85
            self.splitting_ratio = (0.48, 0.52)
            self.loss_db = 0.8
            self.fidelity = 0.92
            self.robustness_score = 0.75
            self.wavelength_response = np.exp(-((np.linspace(1500, 1600, 100) - 1550)**2) / (2 * 20**2))
    
    # 測試評估
    mock_result = MockResult()
    evaluation = quick_evaluate(mock_result, target_splitting_ratio=[0.5, 0.5])
    
    print("=== 設計評估結果 ===")
    print(f"傳輸效率: {evaluation['summary']['transmission']}")
    print(f"插入損耗: {evaluation['summary']['loss_db']} dB")
    print(f"保真度: {evaluation['summary']['fidelity']}")
    print(f"製程容忍度: {evaluation['summary']['robustness']}")
    print(f"綜合評分: {evaluation['summary']['score']}")