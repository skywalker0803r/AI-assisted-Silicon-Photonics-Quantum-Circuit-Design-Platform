"""
Silicon photonics components library
矽光子元件庫
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DesignParameters:
    """設計參數類別"""
    coupling_length: float
    gap: float
    waveguide_width: float
    wavelength: float = 1550e-9  # 預設波長 1550nm
    
class SiliconPhotonicComponent:
    """矽光子元件基類"""
    
    def __init__(self, name: str):
        self.name = name
        self.transmission_matrix = None
        
    def compute_transmission_matrix(self, params: DesignParameters) -> np.ndarray:
        """計算傳輸矩陣"""
        raise NotImplementedError
        
    def get_scattering_parameters(self, params: DesignParameters) -> Dict:
        """獲取散射參數"""
        raise NotImplementedError

class DirectionalCoupler(SiliconPhotonicComponent):
    """方向耦合器 (Directional Coupler)"""
    
    def __init__(self):
        super().__init__("Directional Coupler")
        
    def compute_coupling_coefficient(self, params: DesignParameters) -> float:
        """
        計算耦合係數 (每單位長度)，基於簡化模型。
        kappa 應主要與間距和波導寬度相關，而不是耦合長度本身。
        """
        # 簡化模型：耦合係數隨間距指數衰減
        # 這裡的 np.pi / 20.0 是一個基底耦合強度，可以調整以匹配物理模型
        gap_factor = np.exp(-2 * params.gap / 0.2)  # 指數衰減隨間距
        # 可以根據 waveguide_width 增加額外因子，目前簡化為1
        width_factor = 1.0 
        
        return (np.pi / 20.0) * gap_factor * width_factor
    
    def compute_transmission_matrix(self, params: DesignParameters) -> np.ndarray:
        """
        計算2x2傳輸矩陣
        |t11 t12| |a1|   |b1|
        |t21 t22| |a2| = |b2|
        """
        kappa = self.compute_coupling_coefficient(params)
        L = params.coupling_length
        
        # 耦合模理論傳輸矩陣
        cos_term = np.cos(kappa * L)
        sin_term = 1j * np.sin(kappa * L)
        
        T = np.array([
            [cos_term, sin_term],
            [sin_term, cos_term]
        ])
        
        self.transmission_matrix = T
        return T
    
    def get_splitting_ratio(self, params: DesignParameters) -> Tuple[float, float]:
        """計算分束比"""
        T = self.compute_transmission_matrix(params)
        power_out1 = abs(T[0, 0])**2  # 直通端功率
        power_out2 = abs(T[1, 0])**2  # 耦合端功率
        return power_out1, power_out2

class BeamSplitter(SiliconPhotonicComponent):
    """分束器"""
    
    def __init__(self, target_ratio: float = 0.5):
        super().__init__("Beam Splitter")
        self.target_ratio = target_ratio
        
    def compute_transmission_matrix(self, params: DesignParameters) -> np.ndarray:
        """計算分束器傳輸矩陣"""
        # 使用方向耦合器實現分束器
        coupler = DirectionalCoupler()
        return coupler.compute_transmission_matrix(params)

class PhaseShifter(SiliconPhotonicComponent):
    """相位偏移器"""
    
    def __init__(self, phase: float = 0.0):
        super().__init__("Phase Shifter")
        self.phase = phase
        
    def compute_transmission_matrix(self, params: DesignParameters) -> np.ndarray:
        """計算相位偏移矩陣"""
        return np.array([[np.exp(1j * self.phase)]])

class MachZehnderInterferometer(SiliconPhotonicComponent):
    """馬赫-曾德干涉儀"""
    
    def __init__(self, phase_difference: float = 0.0):
        super().__init__("Mach-Zehnder Interferometer")
        self.phase_difference = phase_difference
        
    def compute_transmission_matrix(self, params: DesignParameters) -> np.ndarray:
        """計算MZI傳輸矩陣"""
        # 兩個50/50分束器 + 相位差
        splitter = BeamSplitter(target_ratio=0.5)
        T_bs = splitter.compute_transmission_matrix(params)
        
        # 中間相位部分
        phase_matrix = np.array([
            [1, 0],
            [0, np.exp(1j * self.phase_difference)]
        ])
        
        # 總傳輸矩陣: T_bs @ phase_matrix @ T_bs
        T_total = T_bs @ phase_matrix @ T_bs
        
        self.transmission_matrix = T_total
        return T_total

class ThreePortInterferometer(SiliconPhotonicComponent):
    """三端口干涉電路 (用於Boson Sampling)"""
    
    def __init__(self):
        super().__init__("Three Port Interferometer")
        
    def compute_transmission_matrix(self, params: DesignParameters) -> np.ndarray:
        """計算3x3傳輸矩陣"""
        # 簡化的3x3干涉矩陣，基於多個方向耦合器組合
        theta1 = params.coupling_length * np.pi / 100  # 歸一化參數
        theta2 = params.gap * np.pi / 2
        theta3 = params.waveguide_width * np.pi
        
        # 構建3x3酉矩陣
        T = np.array([
            [np.cos(theta1), np.sin(theta1) * np.exp(1j*theta2), 0],
            [-np.sin(theta1), np.cos(theta1) * np.exp(1j*theta2), np.sin(theta3)],
            [0, -np.sin(theta3), np.cos(theta3)]
        ])
        
        # 確保酉性（歸一化）
        T = T / np.sqrt(np.sum(np.abs(T)**2, axis=1, keepdims=True))
        
        self.transmission_matrix = T
        return T

def create_component(component_type: str, **kwargs) -> SiliconPhotonicComponent:
    """工廠函數創建元件"""
    components = {
        'directional_coupler': DirectionalCoupler,
        'beam_splitter': BeamSplitter,
        'phase_shifter': PhaseShifter,
        'mach_zehnder': MachZehnderInterferometer,
        'three_port': ThreePortInterferometer
    }
    
    if component_type not in components:
        raise ValueError(f"Unknown component type: {component_type}")
        
    return components[component_type](**kwargs)

# 測試函數
if __name__ == "__main__":
    # 測試方向耦合器
    params = DesignParameters(
        coupling_length=10.0,
        gap=0.2,
        waveguide_width=0.5
    )
    
    coupler = DirectionalCoupler()
    T = coupler.compute_transmission_matrix(params)
    ratio = coupler.get_splitting_ratio(params)
    
    print(f"傳輸矩陣:\n{T}")
    print(f"分束比: {ratio[0]:.3f} / {ratio[1]:.3f}")