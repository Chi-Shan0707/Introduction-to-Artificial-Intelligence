# Bonus 2

这里放的是附加优化代码，主要用于补充 PPT 第 12 页的 MCTS 优化内容。

## 文件

- `train_qlearning_bonus2.py`：优化版 Q-learning。
- `train_mcts_bonus2_optimized.py`：优化版 MCTS。
- `bonus2_supplement_analysis.py`：补充分析脚本。
- `checkpoints/`：Bonus 2 Q-learning 的模型和训练曲线。

## 改过的文件名

为了让 `evaluate.py --agent-class` 能正常导入，整理时改了两个文件名：

- `Train qlearning bonus2.py` → `train_qlearning_bonus2.py`
- `train_mcts_bonus2_optimized.py.py` / `train_mcts_bonus2_PongImproved.py` → `train_mcts_bonus2_optimized.py`

## Q-learning 评测

从 `cartpole_course_project` 目录运行：

```bash
python evaluate.py \
  --agent-class Bonus_2.train_qlearning_bonus2:Agent \
  --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.08,"gamma":0.99,"epsilon":0.0}' \
  --checkpoint Bonus_2/checkpoints/q_learning_bonus2_best.pkl \
  --seed-base 42 --seed-count 100 --max-episode-steps 2000
```

## MCTS

MCTS 不依赖 checkpoint：

```bash
python Bonus_2/train_mcts_bonus2_optimized.py
```
