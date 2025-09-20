# AI 輔助矽光子量子電路設計平台

AI-assisted Silicon Photonics Quantum Circuit Design Platform

## 專案架構

- `core/`: 核心模擬引擎
  - `simulator.py`: 矽光子元件物理模型與量子光學模擬器
  - `components.py`: 元件庫（分束器、相位器、耦合器等）
  - `quantum_sim.py`: 量子態處理與模擬

- `optimization/`: AI優化模組
  - `bayesian_opt.py`: 貝葉斯最佳化
  - `genetic_alg.py`: 遺傳演算法
  - `surrogate_model.py`: 神經網路代理模型
  - `multi_objective.py`: 多目標最佳化

- `evaluation/`: 評估與視覺化
  - `metrics.py`: 性能指標計算
  - `visualization.py`: 結果視覺化
  - `robustness.py`: 製程容忍度分析

- `examples/`: 實例案例
  - `case_a_splitter.py`: 案例A - 50/50分束器設計
  - `case_b_interference.py`: 案例B - 三輸入干涉電路

- `notebooks/`: Jupyter筆記本範例

## 安裝與執行

```bash
pip install -r requirements.txt
python examples/case_a_splitter.py
```