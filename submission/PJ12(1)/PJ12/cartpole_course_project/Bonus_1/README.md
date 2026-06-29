# Bonus 1 目录说明

Bonus 1 用于比较 Q-learning、REINFORCE 和 MCTS 的基础性能、seed 稳定性、计算速率和 S-A 映射。

## 结构

- `evaluate_mcts.py`：MCTS 的主要评测入口。
- `scripts/`：次要脚本，包括批量评测、作图、S-A 映射和测速。
- `results/evaluations/`：四档扰动下三算法 JSON。
- `results/figures/`：报告用图片。
- `results/tables/`：统计表。
- `results/speed/`：计算速率结果。

## 复现命令

```powershell
python .\Bonus_1\scripts\run_bonus1_evals.py
python .\Bonus_1\scripts\plot_bonus1.py
python .\Bonus_1\scripts\sa_mapping_bonus1.py
python .\Bonus_1\scripts\measure_basic_speed.py
```

如需覆盖已有 JSON，可给 `run_bonus1_evals.py` 加 `--force`。
