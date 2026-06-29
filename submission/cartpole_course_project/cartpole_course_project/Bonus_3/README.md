# Bonus 3 目录说明

Bonus 3 聚焦 Q-learning 的鲁棒性改进。主方法是推理端置信度门控邻域 Q 平滑，用来缓解离散状态边界附近的动作跳变。

## 结构

- `train_qlearning_bonus3.py`：主要改进算法代码。
- `scripts/`：次要脚本，包括观测扰动评测、汇总可视化、敏感性实验和失败诊断。
- `results/evaluations/`：初始扰动与观测噪声扰动 JSON。
- `results/figures/`：报告图。
- `results/tables/`：汇总表。
- `results/models/`：训练加噪消融模型。
- `results/failure_diagnostics/`：失败原因诊断。
- `results/sensitivity_smooth/`：平滑维度与强度敏感性实验。

## 最终主方法

- `smooth_mode="confidence"`
- `smooth_dims="angular_only"`
- `smooth_radius=1`
- `smooth_alpha=0.9`
- `confidence_tau=0.8`

## 复现命令

```powershell
python .\Bonus_3\train_qlearning_bonus3.py --train-obs-noise-std 0.005 --output-prefix q_bonus3_obsnoise
python .\Bonus_3\scripts\visualize_bonus3.py
python .\Bonus_3\scripts\sensitivity_smooth_bonus3.py
python .\Bonus_3\scripts\diagnose_failure_states.py
```

其中 `visualize_bonus3.py` 会同时调用初始状态扰动和每步观测噪声扰动评测。
