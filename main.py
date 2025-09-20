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
    
    if args.benchmark or args.all:
        total_count += 1
        if run_benchmark():
            success_count += 1
    
    # 如果沒有指定任何參數，執行快速演示
    if not any([args.demo, args.case_a, args.case_b, args.benchmark, args.all, args.info]):
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