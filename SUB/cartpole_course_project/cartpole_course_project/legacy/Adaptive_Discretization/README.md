# Adaptive Discretization

自适应离散化消融实验。

- `ablation_study.py`：按维度调整 bin 边界
- `../../checkpoints/legacy_adaptive/ablation_results.json`：结果汇总
- `../../checkpoints/legacy_adaptive/*.pkl`：模型
- `../../checkpoints/legacy_adaptive/*.png`：图表

主要结果：

| 配置 | perturb=0 mean | perturb=0.05 mean |
|---|---:|---:|
| no_adapt | 301.05 | 266.22 |
| dim0_pos | 386.22 | 272.56 |
| dim1_vel | 1096.23 | 938.67 |
| dim2_angle | 228.14 | 180.96 |
| dim3_angvel | 1232.45 | 1292.48 |
| all_dims | 320.09 | 241.95 |
