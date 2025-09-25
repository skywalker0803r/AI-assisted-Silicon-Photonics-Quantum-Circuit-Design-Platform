#!/usr/bin/env python3
"""
完整的量子光學演示 - 用最簡單的方式展示矽光子量子電路
"""

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from core.components import DesignParameters, DirectionalCoupler
from core.simulator import CircuitSimulator, create_simple_circuit
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

def simple_quantum_demo():
    """簡單的量子光學演示"""
    print("🌟 歡迎來到量子光學世界！")
    print("=" * 60)
    
    # 1. 創建一個簡單的分束器
    print("\n1. 📡 創建一個簡單的分束器...")
    
    # 這就像一個分束器，可以把光分成兩份
    params = DesignParameters(
        coupling_length=15.7,  # 耦合長度
        gap=0.2,              # 間隙
        waveguide_width=0.5,  # 波導寬度
        wavelength=1550e-9    # 波長
    )
    
    simulator = create_simple_circuit(['directional_coupler'])
    simulator.set_quantum_simulator(n_modes=2, n_photons=1)
    
    # 2. 量子模擬
    print("\n2. ⚛️  進行量子模擬...")
    
    # 放入一個光子，看它會怎麼樣
    quantum_result = simulator.simulate_quantum(params)
    
    print(f"   輸入: 一個光子從左邊進入")
    print(f"   輸出光子分佈: {quantum_result['photon_numbers'][0]:.3f} | {quantum_result['photon_numbers'][1]:.3f}")
    print(f"   量子保真度: {quantum_result['quantum_fidelity']:.3f}")
    
    # 3. 比較不同分束器設計
    print("\n3. 🔬 比較不同分束器設計...")
    
    designs = [
        ("短耦合長度", DesignParameters(8.0, 0.2, 0.5, 1550e-9)),
        ("中等耦合長度", DesignParameters(37.1, 0.2, 0.5, 1550e-9)),
        ("長耦合長度", DesignParameters(37.1+(37.1-8), 0.2, 0.5, 1550e-9)),
    ]
    
    results = []
    for name, design_params in designs:
        result = simulator.simulate_quantum(design_params)
        results.append((name, result))
        
        left_photons = result['photon_numbers'][0]
        right_photons = result['photon_numbers'][1]
        
        print(f"   {name:8s}: 左邊 {left_photons:.3f} 個光子, 右邊 {right_photons:.3f} 個光子")
    
    # 4. 視覺化模擬結果
    print("\n4. 📊 視覺化模擬結果...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 光子分佈圖
    names = [r[0] for r in results]
    left_counts = [r[1]['photon_numbers'][0] for r in results]
    right_counts = [r[1]['photon_numbers'][1] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax1.bar(x - width/2, left_counts, width, label='左邊輸出', color='blue', alpha=0.7)
    ax1.bar(x + width/2, right_counts, width, label='右邊輸出', color='red', alpha=0.7)
    ax1.set_xlabel('耦合長度設計')
    ax1.set_ylabel('光子數量期望值')
    ax1.set_title('不同耦合長度設計的光子分配')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 保真度比較
    fidelities = [r[1]['quantum_fidelity'] for r in results]
    ax2.bar(names, fidelities, color='green', alpha=0.7)
    ax2.set_xlabel('耦合長度設計')
    ax2.set_ylabel('量子保真度')
    ax2.set_title('量子保真度比較')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 5. 解釋物理原理
    print("\n5. 🎓 這些數值代表什麼？")
    print("   - 光子數量期望值: 表示光子有多少機率會出現在左邊或右邊的輸出埠")
    print("   - 量子保真度: 表示我們的分束器在實現量子操作上的準確性")
    print("   - 越接近 0.5/0.5 分配 = 越完美的50/50分束器")
    print("   - 保真度越接近 1.0 = 量子操作效果越好")

def advanced_quantum_demo():
    """進階量子效果演示"""
    print("\n" + "=" * 60)
    print("🚀 進階量子模擬演示")
    print("=" * 60)
    
    # 創建理想的50/50分束器
    print("\n1. 🎯 尋找最佳分束器設計...")
    
    # 掃描不同的耦合長度
    lengths = np.linspace(5, 50, 50)
    quantum_results = []
    
    simulator = create_simple_circuit(['directional_coupler'])
    simulator.set_quantum_simulator(n_modes=2, n_photons=1)
    
    for length in lengths:
        params = DesignParameters(length, 0.2, 0.5, 1550e-9)
        result = simulator.simulate_quantum(params)
        quantum_results.append(result)
    
    # 找到最接近50/50的設計
    splitting_errors = []
    for result in quantum_results:
        left, right = result['photon_numbers']
        error = abs(left - 0.5) + abs(right - 0.5)
        splitting_errors.append(error)
    
    best_idx = np.argmin(splitting_errors)
    best_length = lengths[best_idx]
    best_result = quantum_results[best_idx]
    
    print(f"   最佳耦合長度: {best_length:.2f} μm")
    print(f"   光子分配: {best_result['photon_numbers'][0]:.3f} / {best_result['photon_numbers'][1]:.3f}")
    print(f"   量子保真度: {best_result['quantum_fidelity']:.3f}")
    
    # 繪製優化曲線
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    left_photons = [r['photon_numbers'][0] for r in quantum_results]
    right_photons = [r['photon_numbers'][1] for r in quantum_results]
    plt.plot(lengths, left_photons, 'b-o', label='左邊輸出')
    plt.plot(lengths, right_photons, 'r-s', label='右邊輸出')
    plt.axhline(y=0.5, color='green', linestyle='--', label='理想50%')
    plt.axvline(x=best_length, color='purple', linestyle=':', label=f'最佳耦合長度 {best_length:.1f}μm')
    plt.xlabel('耦合長度 (μm)')
    plt.ylabel('光子數量期望值')
    plt.title('光子分配 vs 耦合長度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    fidelities = [r['quantum_fidelity'] for r in quantum_results]
    plt.plot(lengths, fidelities, 'g-^')
    plt.axvline(x=best_length, color='purple', linestyle=':', label=f'最佳耦合長度')
    plt.xlabel('耦合長度 (μm)')
    plt.ylabel('量子保真度')
    plt.title('量子保真度 vs 耦合長度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(lengths, splitting_errors, 'm-d')
    plt.axvline(x=best_length, color='purple', linestyle=':', label=f'最佳點')
    plt.xlabel('耦合長度 (μm)')
    plt.ylabel('分束比誤差')
    plt.title('50/50分束比誤差')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # 顯示最佳設計的量子態分佈
    # 直接使用 best_result 中的光子數量期望值
    n1_expect = best_result['photon_numbers'][0]
    n2_expect = best_result['photon_numbers'][1]
    
    states_labels = ['模式1 (1,0)', '模式2 (0,1)'] # 更具描述性的標籤
    probabilities = [n1_expect, n2_expect]
    
    plt.bar(states_labels, probabilities, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('光子數量期望值') # 更改標籤以反映期望值
    plt.title('最佳設計的光子分佈') # 更具體的標題
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    # 運行所有演示
    simple_quantum_demo()
    advanced_quantum_demo() 

if __name__ == "__main__":
    main()