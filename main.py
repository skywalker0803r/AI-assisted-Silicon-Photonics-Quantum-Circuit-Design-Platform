#!/usr/bin/env python3
"""
AI輔助矽光子量子電路設計平台 - 主執行腳本
Main execution script for AI-assisted Silicon Photonics Quantum Circuit Design Platform
"""

import argparse
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.rc('font', family='Microsoft JhengHei')

def setup_environment():
    """設置執行環境"""
    # 添加當前目錄到Python路徑
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        # 測試核心模組導入
        from core.components import DirectionalCoupler
        from core.simulator import CircuitSimulator
        from optimization.bayesian_opt import BayesianOptimizer
        from evaluation.metrics import MetricsCalculator
        print("✅ 核心模組導入成功")
        return True
    except ImportError as e:
        print(f"❌ 模組導入失敗: {e}")
        print("請檢查模組是否正確安裝")
        print("提示：某些進階功能需要額外套件，但基本功能應該能正常運作")
        return False

def run_case_a():
    """執行案例A：50/50分束器設計"""
    print("\n" + "="*60)
    print("執行案例A：高保真50/50分束器設計")
    print("="*60)
    
    try:
        from examples.case_a_splitter import main as case_a_main
        case_a_main()
        return True
    except Exception as e:
        print(f"❌ 案例A執行失敗: {e}")
        return False

def run_case_b():
    """執行案例B：三輸入干涉電路設計"""
    print("\n" + "="*60)
    print("執行案例B：三輸入干涉電路設計")
    print("="*60)
    
    try:
        from examples.case_b_interference import main as case_b_main
        case_b_main()
        return True
    except Exception as e:
        print(f"❌ 案例B執行失敗: {e}")
        return False

def run_quick_demo():
    """執行快速演示"""
    print("\n" + "="*60)
    print("快速演示：基本功能測試")
    print("="*60)
    
    try:
        from core.components import DesignParameters, DirectionalCoupler
        from core.simulator import create_simple_circuit
        from evaluation.metrics import quick_evaluate
        
        # 創建設計參數
        params = DesignParameters(
            coupling_length=15.0,
            gap=0.2,
            waveguide_width=0.5,
            wavelength=1550e-9
        )
        
        print("1. 測試方向耦合器...")
        coupler = DirectionalCoupler()
        T = coupler.compute_transmission_matrix(params)
        ratio = coupler.get_splitting_ratio(params)
        
        print(f"   分束比: {ratio[0]:.3f} / {ratio[1]:.3f}")
        print(f"   總功率: {sum(ratio):.3f}")
        
        print("2. 測試電路模擬...")
        simulator = create_simple_circuit(['directional_coupler'])
        result = simulator.simulate_classical(params)
        
        print(f"   傳輸效率: {result.transmission_efficiency:.3f}")
        print(f"   損耗: {result.loss_db:.2f} dB")
        print(f"   保真度: {result.fidelity:.3f}")
        
        print("3. 測試性能評估...")
        evaluation = quick_evaluate(result, target_splitting_ratio=[0.5, 0.5])
        print(f"   綜合評分: {evaluation['composite_score']:.3f}")
        
        print("✅ 快速演示完成！所有基本功能正常運作。")
        return True
        
    except Exception as e:
        print(f"❌ 快速演示失敗: {e}")
        return False

def run_case_c():
    """執行案例C：量子模擬演示"""
    print("\n" + "="*60)
    print("執行案例C：量子模擬演示")
    print("="*60)
    
    try:
        from quantum_demo import main as quantum_demo_main
        quantum_demo_main()
        return True
    except Exception as e:
        print(f"❌ 案例C執行失敗: {e}")
        return False

def run_case_d():
    """執行案例D：使用量子模擬進行分束器設計"""
    print("\n" + "="*60)
    print("執行案例D：使用量子模擬進行分束器設計")
    print("="*60)
    print("⚠️  警告：此模式使用量子模擬進行最佳化，速度會非常慢。")
    
    try:
        from examples.case_a_splitter import SplitterDesignOptimizer
        from core.components import DesignParameters
        from typing import Dict
        import matplotlib.pyplot as plt

        class CaseDOptimizer(SplitterDesignOptimizer):
            """使用量子模擬的50/50分束器設計最佳化器"""
            
            def objective_function(self, params: Dict[str, float]) -> float:
                """
                目標函數：基於 simulate_quantum 的結果
                """
                design_params = DesignParameters(
                    coupling_length=params['coupling_length'],
                    gap=params['gap'],
                    waveguide_width=params['waveguide_width'],
                    wavelength=1550e-9
                )
                
                try:
                    quantum_result = self.simulator.simulate_quantum(design_params)
                    photon_numbers = quantum_result['photon_numbers']
                    quantum_fidelity = quantum_result['quantum_fidelity']
                    splitting_error = abs(photon_numbers[0] - 0.5)
                    composite_score = quantum_fidelity - splitting_error
                    
                    if not hasattr(self, 'optimization_history'):
                        self.optimization_history = []
                    
                    self.optimization_history.append({
                        'params': params.copy(),
                        'quantum_fidelity': quantum_fidelity,
                        'splitting_error': splitting_error,
                        'objective': composite_score,
                        'photon_numbers': photon_numbers
                    })
                    
                    return composite_score
                    
                except Exception as e:
                    return -1.0

        optimizer = CaseDOptimizer()
        n_iterations = 100 
        print(f"開始最佳化，迭代次數: {n_iterations}")
        result = optimizer.run_optimization(n_iterations=n_iterations)

        print(f"\n=== 案例D 最佳化結果分析 ====")
        print(f"最佳化時間: {result['optimization_time']:.2f} 秒")
        print(f"\n最佳設計參數:")
        for param, value in result['params'].items():
            print(f"  {param}: {value:.4f}")

        if optimizer.optimization_history:
            best_entry = max(optimizer.optimization_history, key=lambda x: x['objective'])
            print(f"\n性能指標 (基於量子模擬):")
            print(f"  綜合評分: {best_entry['objective']:.4f}")
            print(f"  輸出光子分佈: {best_entry['photon_numbers'][0]:.4f} / {best_entry['photon_numbers'][1]:.4f}")
            print(f"  量子保真度: {best_entry['quantum_fidelity']:.4f}")

            print("\n📊 正在繪製結果圖表...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('案例D：量子最佳化過程全方位分析', fontsize=16)

            # 1. 最佳化收斂過程
            iterations = range(len(optimizer.optimization_history))
            scores = [entry['objective'] for entry in optimizer.optimization_history]
            ax1.plot(iterations, scores, 'b-', marker='o', label='目標評分')
            ax1.set_xlabel('迭代次數')
            ax1.set_ylabel('綜合評分 (保真度 - 誤差)')
            ax1.set_title('1. 最佳化收斂過程')
            ax1.legend()
            ax1.grid(True, alpha=0.5)

            # 2. 性能指標演化
            fidelities = [entry['quantum_fidelity'] for entry in optimizer.optimization_history]
            errors = [entry['splitting_error'] for entry in optimizer.optimization_history]
            ax2.plot(iterations, fidelities, 'g-', marker='.', label='量子保真度')
            ax2.plot(iterations, errors, 'r-', marker='.', label='分束誤差')
            ax2.set_xlabel('迭代次數')
            ax2.set_ylabel('指標值')
            ax2.set_title('2. 性能指標演化')
            ax2.legend()
            ax2.grid(True, alpha=0.5)

            # 3. 參數空間探索
            coupling_lengths = [entry['params']['coupling_length'] for entry in optimizer.optimization_history]
            gaps = [entry['params']['gap'] for entry in optimizer.optimization_history]
            scatter = ax3.scatter(coupling_lengths, gaps, c=scores, cmap='viridis', alpha=0.8)
            ax3.set_xlabel('耦合長度 (μm)')
            ax3.set_ylabel('間距 (μm)')
            ax3.set_title('3. 參數空間探索')
            fig.colorbar(scatter, ax=ax3, label='綜合評分')
            ax3.grid(True, alpha=0.3)

            # 4. 最佳結果的光子分佈
            best_photon_numbers = best_entry['photon_numbers']
            ports = ['端口 1', '端口 2']
            ax4.bar(ports, best_photon_numbers, color=['skyblue', 'salmon'])
            ax4.set_ylabel('光子數期望值')
            ax4.set_title(f'4. 最佳設計光子分佈 (評分: {best_entry["objective"]:.3f})')
            ax4.set_ylim(0, 1.0)
            for i, v in enumerate(best_photon_numbers):
                ax4.text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
            ax4.grid(True, axis='y', alpha=0.5)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        else:
            print("\n無法分析或繪製量子結果，因為最佳化歷史為空。")

        return True

    except Exception as e:
        print(f"❌ 案例D執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_benchmark():
    """執行性能基準測試"""
    print("\n" + "="*60)
    print("性能基準測試")
    print("="*60)
    
    try:
        from core.components import DesignParameters
        from core.simulator import create_simple_circuit
        from optimization.bayesian_opt import optimize_design
        
        # 測試模擬速度
        print("1. 測試模擬速度...")
        simulator = create_simple_circuit(['directional_coupler'])
        
        params = DesignParameters(15.0, 0.2, 0.5, 1550e-9)
        
        start_time = time.time()
        n_simulations = 100
        
        for _ in range(n_simulations):
            result = simulator.simulate_classical(params)
        
        sim_time = time.time() - start_time
        print(f"   {n_simulations}次模擬耗時: {sim_time:.3f}秒")
        print(f"   平均每次模擬: {sim_time/n_simulations*1000:.2f}毫秒")
        
        # 測試最佳化速度
        print("2. 測試最佳化速度...")
        
        def test_objective(params):
            design_params = DesignParameters(
                params['coupling_length'], params['gap'], 
                params['waveguide_width'], 1550e-9
            )
            result = simulator.simulate_classical(design_params)
            return result.transmission_efficiency
        
        bounds = {
            'coupling_length': (10.0, 20.0),
            'gap': (0.15, 0.25),
            'waveguide_width': (0.45, 0.55)
        }
        
        start_time = time.time()
        best_params, best_value, history = optimize_design(
            test_objective, bounds, n_iterations=10, verbose=False
        )
        opt_time = time.time() - start_time
        
        print(f"   10次迭代最佳化耗時: {opt_time:.3f}秒")
        print(f"   最佳結果: {best_value:.4f}")
        
        print("✅ 性能基準測試完成！")
        return True
        
    except Exception as e:
        print(f"❌ 性能基準測試失敗: {e}")
        return False

def show_system_info():
    """顯示系統資訊"""
    print("\n" + "="*60)
    print("系統資訊")
    print("="*60)
    
    import platform
    import numpy as np
    
    print(f"Python版本: {platform.python_version()}")
    print(f"作業系統: {platform.system()} {platform.release()}")
    print(f"NumPy版本: {np.__version__}")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        print("PyTorch: 未安裝")
    
    try:
        import qutip
        print(f"QuTiP版本: {qutip.__version__}")
    except ImportError:
        print("QuTiP: 未安裝")
    
    try:
        import sklearn
        print(f"scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn: 未安裝")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AI輔助矽光子量子電路設計平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python main.py --demo          # 執行快速演示
  python main.py --case-a        # 執行案例A
  python main.py --case-b        # 執行案例B
  python main.py --case-c        # 執行案例C：量子模擬演示
  python main.py --case-d        # 執行案例D：使用量子模擬進行最佳化 (速度慢)
  python main.py --benchmark     # 執行性能測試
  python main.py --all           # 執行所有案例
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='執行快速演示')
    parser.add_argument('--case-a', action='store_true', 
                       help='執行案例A：50/50分束器設計')
    parser.add_argument('--case-b', action='store_true', 
                       help='執行案例B：三輸入干涉電路設計')
    parser.add_argument('--case-c', action='store_true', 
                       help='執行案例C：量子模擬演示')
    parser.add_argument('--case-d', action='store_true', 
                       help='執行案例D：使用量子模擬進行最佳化 (速度慢)')
    parser.add_argument('--benchmark', action='store_true', 
                       help='執行性能基準測試')
    parser.add_argument('--all', action='store_true', 
                       help='執行所有案例')
    parser.add_argument('--info', action='store_true', 
                       help='顯示系統資訊')
    
    args = parser.parse_args()
    
    # 顯示歡迎訊息
    print("🌟 AI輔助矽光子量子電路設計平台")
    print("AI-assisted Silicon Photonics Quantum Circuit Design Platform")
    print("版本: 1.0.0")
    
    # 設置環境
    if not setup_environment():
        sys.exit(1)
    
    # 顯示系統資訊
    if args.info:
        show_system_info()
        return
    
    # 執行功能
    success_count = 0
    total_count = 0
    
    if args.demo or args.all:
        total_count += 1
        if run_quick_demo():
            success_count += 1
    
    if args.case_a or args.all:
        total_count += 1
        if run_case_a():
            success_count += 1
    
    if args.case_b or args.all:
        total_count += 1
        if run_case_b():
            success_count += 1

    if args.case_c or args.all:
        total_count += 1
        if run_case_c():
            success_count += 1

    if args.case_d or args.all:
        total_count += 1
        if run_case_d():
            success_count += 1
    
    if args.benchmark or args.all:
        total_count += 1
        if run_benchmark():
            success_count += 1
    
    # 如果沒有指定任何參數，執行快速演示
    if not any([args.demo, args.case_a, args.case_b, args.case_c, args.case_d, args.benchmark, args.all, args.info]):
        total_count += 1
        if run_quick_demo():
            success_count += 1
    
    # 顯示總結
    if total_count > 0:
        print(f"\n" + "="*60)
        print(f"執行總結: {success_count}/{total_count} 項目成功完成")
        if success_count == total_count:
            print("🎉 所有項目都成功執行！")
        else:
            print("⚠️  部分項目執行失敗，請檢查錯誤訊息")
        print("="*60)

if __name__ == "__main__":
    main()