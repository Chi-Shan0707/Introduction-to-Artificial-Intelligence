

## 1. 基础任务

- `train_qlearning.py`
- `train_reinforce.py`
- `train_mcts.py`
- `evaluate.py`
- `vis.py`
- `vis_mcts.py`

Q-learning 和 REINFORCE 的模型保存在 `checkpoints/`。

## 2. Bonus 1

结构：

- `Bonus_1/evaluate_mcts.py`：MCTS 评测入口，属于 Bonus 1 的主要代码。
- `Bonus_1/scripts/`：批量评测、作图、S-A 映射、测速等次要脚本。
- `Bonus_1/results/evaluations/`：三算法四档扰动 JSON。
- `Bonus_1/results/figures/`：报告图。
- `Bonus_1/results/tables/`：定量统计表。
- `Bonus_1/results/speed/`：基础计算速率结果。

主复现顺序：

```
python .\Bonus_1\scripts\run_bonus1_evals.py
python .\Bonus_1\scripts\plot_bonus1.py
python .\Bonus_1\scripts\sa_mapping_bonus1.py
python .\Bonus_1\scripts\measure_basic_speed.py
```

## 3. Bonus 3

结构：

- `Bonus_3/train_qlearning_bonus3.py`：Bonus 3 的主要改进算法代码。
- `Bonus_3/scripts/`：观测扰动评测、汇总可视化、敏感性实验、失败诊断脚本。
- `Bonus_3/results/evaluations/`：初始扰动与观测噪声扰动 JSON。
- `Bonus_3/results/figures/`：鲁棒性曲线、箱线图、训练曲线。
- `Bonus_3/results/tables/`：四档扰动汇总表。
- `Bonus_3/results/models/`：训练加噪消融模型。
- `Bonus_3/results/failure_diagnostics/`：失败原因诊断。
- `Bonus_3/results/sensitivity_smooth/`：平滑维度与强度敏感性实验。

最终主方法为推理端置信度门控邻域 Q 平滑：

- `smooth_mode="confidence"`
- `smooth_dims="angular_only"`
- `smooth_radius=1`
- `smooth_alpha=0.9`
- `confidence_tau=0.8`

主复现顺序：

```
python .\Bonus_3\train_qlearning_bonus3.py --train-obs-noise-std 0.005 --output-prefix q_bonus3_obsnoise
python .\Bonus_3\scripts\visualize_bonus3.py
python .\Bonus_3\scripts\sensitivity_smooth_bonus3.py
python .\Bonus_3\scripts\diagnose_failure_states.py
