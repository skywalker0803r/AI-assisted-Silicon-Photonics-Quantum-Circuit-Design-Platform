"""
Silicon photonics quantum circuit simulator
矽光子量子電路模擬器
"""
import numpy as np
import qutip as qt
HAS_QUTIP = True
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import random # Add this import for random number generation

from .components import SiliconPhotonicComponent, DesignParameters

@dataclass
class SimulationResult:
    """模擬結果數據結構"""
    transmission_efficiency: float
    splitting_ratio: Tuple[float, ...]
    fidelity: float
    loss_db: float
    phase_response: np.ndarray
    wavelength_response: np.ndarray
    robustness_score: float

class QuantumStateSimulator:
    """量子態模擬器"""
    
    def __init__(self, n_modes: int = 2, n_photons: int = 1):
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.fock_dim = n_photons + 1
        
    def create_fock_state(self, photon_numbers: List[int]):
        """創建Fock態"""
        if len(photon_numbers) != self.n_modes:
            raise ValueError("Photon numbers must match number of modes")
            
        # 使用QuTiP創建張量積Fock態
        states = []
        for n in photon_numbers:
            states.append(qt.fock(self.fock_dim, n))
        return qt.tensor(states)
    

    
    def calculate_fidelity(self, state1, state2) -> float:
        """計算態保真度"""
        return abs(qt.fidelity(state1, state2))**2

class CircuitSimulator:
    """電路模擬器主類"""
    
    def __init__(self):
        self.components = []
        self.quantum_sim = None
        
    def add_component(self, component: SiliconPhotonicComponent):
        """添加元件到電路"""
        self.components.append(component)
        
    def set_quantum_simulator(self, n_modes: int, n_photons: int = 1):
        """設置量子模擬器"""
        self.quantum_sim = QuantumStateSimulator(n_modes, n_photons)
    
    def simulate_classical(self, params: DesignParameters, 
                         wavelengths: Optional[np.ndarray] = None, 
                         run_wavelength_sweep: bool = True, 
                         run_robustness_check: bool = True) -> SimulationResult:
        """經典光學模擬"""
        if wavelengths is None:
            wavelengths = np.linspace(1500e-9, 1600e-9, 100)
            
        # 計算總傳輸矩陣
        total_matrix = np.eye(2, dtype=complex)
        
        for component in self.components:
            T = component.compute_transmission_matrix(params)
            if T.shape[0] == total_matrix.shape[1]:
                total_matrix = T @ total_matrix
                
        # 計算性能指標
        transmission_eff = np.abs(total_matrix[0, 0])**2
        
        # 分束比計算
        power_outputs = []
        for i in range(total_matrix.shape[0]):
            power_outputs.append(np.abs(total_matrix[i, 0])**2)
        splitting_ratio = tuple(power_outputs)
        
        # 損耗計算 (dB)
        total_power = sum(power_outputs)
        loss_db = -10 * np.log10(total_power) if total_power > 0 else float('inf')
        
        # 波長響應模擬 (可選)
        if run_wavelength_sweep:
            wavelength_response = self._simulate_wavelength_response(params, wavelengths)
        else:
            wavelength_response = np.array([]) # 如果不執行，返回空陣列
        
        # 相位響應
        phase_response = np.angle(total_matrix[0, 0])
        
        # 簡化保真度計算（與理想設計比較）
        ideal_matrix = self._get_ideal_matrix(total_matrix.shape[0]) # 傳入模式數
        fidelity = self._calculate_matrix_fidelity(total_matrix, ideal_matrix)
        
        # 評估製程容忍度 (可選)
        if run_robustness_check:
            robustness_score = self._assess_robustness(params)
        else:
            robustness_score = 0.0 # 如果不執行，返回0.0
        
        return SimulationResult(
            transmission_efficiency=transmission_eff,
            splitting_ratio=splitting_ratio,
            fidelity=fidelity,
            loss_db=loss_db,
            phase_response=np.array([phase_response]),
            wavelength_response=wavelength_response,
            robustness_score=robustness_score
        )
    
    def simulate_quantum(self, params: DesignParameters, 
                        input_state: Optional = None) -> Dict:
        """完整量子模擬"""
        if self.quantum_sim is None:
            raise ValueError("Quantum simulator not initialized")
            
        if input_state is None:
            # 預設單光子輸入態 |1,0⟩
            input_state = self.quantum_sim.create_fock_state([1, 0])
        
        # 獲取經典傳輸矩陣
        total_matrix = np.eye(2, dtype=complex)
        for component in self.components:
            T = component.compute_transmission_matrix(params)
            if T.shape[0] == total_matrix.shape[1]:
                total_matrix = T @ total_matrix
        
        # 假設 input_state 是單光子態，例如 |1,0> (1 photon in mode 1, 0 in mode 2)
        # 根據 total_matrix 的元素直接構建輸出疊加態
        fock_dim = self.quantum_sim.fock_dim
        
        # 定義單光子雙模式的基態
        state_10 = qt.tensor(qt.fock(fock_dim, 1), qt.fock(fock_dim, 0))
        state_01 = qt.tensor(qt.fock(fock_dim, 0), qt.fock(fock_dim, 1))
        
        # 假設輸入態是 |1,0> (即 input_state == state_10)
        # 輸出態將是 total_matrix[0,0] * |1,0> + total_matrix[1,0] * |0,1>
        # 這裡 total_matrix[0,0] 是從模式1到模式1的振幅
        # total_matrix[1,0] 是從模式1到模式2的振幅
        output_state = total_matrix[0, 0] * state_10 + total_matrix[1, 0] * state_01
        
        # 確保輸出態經過歸一化
        output_state = output_state.unit()
        
        # 計算輸出光子數期望值
        a1 = qt.tensor(qt.destroy(self.quantum_sim.fock_dim), qt.qeye(self.quantum_sim.fock_dim))
        a2 = qt.tensor(qt.qeye(self.quantum_sim.fock_dim), qt.destroy(self.quantum_sim.fock_dim))
        
        n1_expect = qt.expect(a1.dag() * a1, output_state)
        n2_expect = qt.expect(a2.dag() * a2, output_state)
        
        # 理想50/50分束器的輸出態 (考慮到 DirectionalCoupler 引入的 pi/2 相位)
        ideal_superposition = (self.quantum_sim.create_fock_state([1, 0]) + 
                              1j * self.quantum_sim.create_fock_state([0, 1])).unit()
        
        quantum_fidelity = self.quantum_sim.calculate_fidelity(output_state, ideal_superposition)
        
        return {
            'input_state': input_state,
            'output_state': output_state,
            'photon_numbers': [float(n1_expect), float(n2_expect)],
            'quantum_fidelity': quantum_fidelity,
            'classical_matrix': total_matrix
        }
    
    def _simulate_wavelength_response(self, params: DesignParameters, 
                                    wavelengths: np.ndarray) -> np.ndarray:
        """模擬波長響應"""
        response = []
        original_wavelength = params.wavelength
        
        for wl in wavelengths:
            params.wavelength = wl
            # 重新計算所有元件的傳輸矩陣
            total_matrix = np.eye(2, dtype=complex)
            for component in self.components:
                T = component.compute_transmission_matrix(params)
                if T.shape[0] == total_matrix.shape[1]:
                    total_matrix = T @ total_matrix
            response.append(np.abs(total_matrix[0, 0])**2)
            
        params.wavelength = original_wavelength  # 恢復原始波長
        return np.array(response)
    
    def _get_ideal_matrix(self, n_modes: int) -> np.ndarray:
        """
        獲取理想傳輸矩陣，根據模式數量動態生成。
        對於2模式，返回50/50分束器矩陣。
        對於3模式，返回一個簡化的理想玻色採樣矩陣。
        """
        if n_modes == 2:
            # 對於50/50分束器的理想矩陣
            return np.array([[1/np.sqrt(2), 1j/np.sqrt(2)],
                            [1j/np.sqrt(2), 1/np.sqrt(2)]])
        elif n_modes == 3:
            # 對於3模式玻色採樣的簡化理想矩陣 (例如一個Haar隨機酉矩陣)
            # 這裡使用 case_b_interference.py 中定義的理想矩陣
            return np.array([
                [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                [1/np.sqrt(3), 1/np.sqrt(3) * np.exp(1j*2*np.pi/3), 1/np.sqrt(3) * np.exp(1j*4*np.pi/3)],
                [1/np.sqrt(3), 1/np.sqrt(3) * np.exp(1j*4*np.pi/3), 1/np.sqrt(3) * np.exp(1j*2*np.pi/3)]
            ])
        else:
            raise ValueError(f"Unsupported number of modes for ideal matrix: {n_modes}")
    
    def _calculate_matrix_fidelity(self, actual: np.ndarray, ideal: np.ndarray) -> float:
        """計算矩陣保真度"""
        if actual.shape != ideal.shape:
            # 如果形狀不匹配，直接返回0.0，表示完全不匹配
            return 0.0
        
        # 使用矩陣的Frobenius內積計算相似度
        fidelity = np.abs(np.trace(np.conj(actual).T @ ideal))**2
        fidelity /= (np.trace(np.conj(actual).T @ actual) * np.trace(np.conj(ideal).T @ ideal))
        return float(np.real(fidelity))
    
    def _assess_robustness(self, nominal_params: DesignParameters, 
                           perturbation_percentage: float = 0.02, # 2%的參數擾動
                           num_samples: int = 10) -> float:
        """
        評估製程容忍度：通過對設計參數進行小幅擾動，模擬多次並評估性能變化。
        穩健性分數越高，表示設計對參數變化越不敏感。
        """
        perturbed_fidelities = []
        perturbed_losses = []
        
        # 儲存原始參數，以便恢復
        original_coupling_length = nominal_params.coupling_length
        original_gap = nominal_params.gap
        original_waveguide_width = nominal_params.waveguide_width

        # 獲取元件的模式數量 (假設所有元件模式數相同)
        n_modes = self.components[0].compute_transmission_matrix(nominal_params).shape[0]

        for _ in range(num_samples):
            # 產生擾動參數
            perturbed_params = DesignParameters(
                coupling_length=nominal_params.coupling_length * (1 + random.uniform(-perturbation_percentage, perturbation_percentage)),
                gap=nominal_params.gap * (1 + random.uniform(-perturbation_percentage, perturbation_percentage)),
                waveguide_width=nominal_params.waveguide_width * (1 + random.uniform(-perturbation_percentage, perturbation_percentage)),
                wavelength=nominal_params.wavelength # 波長通常不擾動
            )
            
            # 模擬擾動後的性能
            total_matrix = np.eye(n_modes, dtype=complex) # 使用正確的模式數初始化矩陣
            for component in self.components:
                T = component.compute_transmission_matrix(perturbed_params)
                if T.shape[0] == total_matrix.shape[1]:
                    total_matrix = T @ total_matrix
            
            # 計算損耗
            power_outputs = []
            for i in range(total_matrix.shape[0]):
                power_outputs.append(np.abs(total_matrix[i, 0])**2)
            total_power = sum(power_outputs)
            loss_db = -10 * np.log10(total_power) if total_power > 0 else float('inf')
            
            # 計算保真度 (使用動態的理想矩陣)
            ideal_matrix = self._get_ideal_matrix(n_modes) # 傳入模式數
            fidelity = self._calculate_matrix_fidelity(total_matrix, ideal_matrix)
            
            perturbed_fidelities.append(fidelity)
            perturbed_losses.append(loss_db)
        
        # 恢復原始參數
        nominal_params.coupling_length = original_coupling_length
        nominal_params.gap = original_gap
        nominal_params.waveguide_width = original_waveguide_width

        # 評估穩健性：高平均保真度，低保真度標準差
        avg_fidelity = np.mean(perturbed_fidelities)
        std_fidelity = np.std(perturbed_fidelities)
        
        # 穩健性分數：例如 avg_fidelity - 2 * std_fidelity，並限制在0到1之間
        robustness_score = np.clip(avg_fidelity - 2 * std_fidelity, 0.0, 1.0)
        
        return float(robustness_score)

# 輔助函數
def create_simple_circuit(component_types: List[str]) -> CircuitSimulator:
    """創建簡單電路"""
    from .components import create_component
    
    simulator = CircuitSimulator()
    
    for comp_type in component_types:
        component = create_component(comp_type)
        simulator.add_component(component)
    
    return simulator

# 測試函數
if __name__ == "__main__":
    # 測試分束器電路
    params = DesignParameters(
        coupling_length=15.0,
        gap=0.2,
        waveguide_width=0.5
    )
    
    simulator = create_simple_circuit(['directional_coupler'])
    simulator.set_quantum_simulator(n_modes=2, n_photons=1)
    
    result = simulator.simulate_classical(params)
    
    print("=== 模擬結果 ===")
    print(f"傳輸效率: {result.transmission_efficiency:.3f}")
    print(f"分束比: {result.splitting_ratio}")
    print(f"損耗: {result.loss_db:.2f} dB")
    print(f"保真度: {result.fidelity:.3f}")
    print(f"製程容忍度: {result.robustness_score:.3f}")