# 复现与提交结构说明

本文档说明本项目中用于复现报告结果的主代码、次要脚本和结果目录。

## 1. 基础任务

基础任务保持老师给定的主结构：

- `train_qlearning.py`
- `train_reinforce.py`
- `train_mcts.py`
- `evaluate.py`
- `vis.py`
- `vis_mcts.py`

Q-learning 和 REINFORCE 的模型保存在 `checkpoints/`；MCTS 是在线规划算法，没有 checkpoint。

## 2. Bonus 1

结构：

- `Bonus_1/evaluate_mcts.py`：MCTS 评测入口，属于 Bonus 1 的主要代码。
- `Bonus_1/scripts/`：批量评测、作图、S-A 映射、测速等次要脚本。
- `Bonus_1/results/evaluations/`：三算法四档扰动 JSON。
- `Bonus_1/results/figures/`：报告图。
- `Bonus_1/results/tables/`：定量统计表。
- `Bonus_1/results/speed/`：基础计算速率结果。

主复现顺序：

```powershell
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

```powershell
python .\Bonus_3\train_qlearning_bonus3.py --train-obs-noise-std 0.005 --output-prefix q_bonus3_obsnoise
python .\Bonus_3\scripts\visualize_bonus3.py
python .\Bonus_3\scripts\sensitivity_smooth_bonus3.py
python .\Bonus_3\scripts\diagnose_failure_states.py
```

## 4. 快捷命令

常用命令统一放在项目根目录外侧的 `Quick_Command` 文件中。建议从 `PJ12` 根目录执行，再进入 `cartpole_course_project`。

## 5. 关于禁改文件

`vis_mcts.py` 做过一处导入修正：将不存在的 `train_mcts_backup` 改为本次提交实际使用的 `train_mcts`。该问题已经在 `ISSUE_NOTE.txt` 中说明。
