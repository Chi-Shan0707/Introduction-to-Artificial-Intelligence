# Agents

## Checkpoints

| Folder | Content |
|---|---|
| `checkpoints/base/` | 基础 Q-learning、REINFORCE |
| `checkpoints/bonus2/` | Bonus 2 优化 Q-learning |
| `checkpoints/bonus3/` | Bonus 3 观测噪声训练模型 |
| `checkpoints/legacy_adaptive/` | 自适应离散化消融 |

MCTS 无 checkpoint。

## Base

```bash
python evaluate.py --agent-class train_qlearning:Agent --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}' --checkpoint checkpoints/base/q_learning_model.pkl --seed-base 42 --seed-count 100 --max-episode-steps 2000
```

```bash
python evaluate.py --agent-class train_reinforce:Agent --agent-init-kwargs '{"n_state":4,"hidden_c":16,"n_action":2}' --checkpoint checkpoints/base/reinforce_model.pt --seed-base 42 --seed-count 100 --max-episode-steps 2000
```

```bash
python Bonus_1/evaluate_mcts.py --seed-base 42 --seed-count 100 --iteration-budget 80 --lookahead-target 200 --start-cp 200 --max-episode-steps 2000
```

## Bonus 2

```bash
python evaluate.py --agent-class Bonus_2.train_qlearning_bonus2:Agent --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.08,"gamma":0.99,"epsilon":0.0}' --checkpoint checkpoints/bonus2/q_learning_bonus2_best.pkl --seed-base 42 --seed-count 100 --max-episode-steps 2000
```

```bash
python Bonus_2/train_mcts_bonus2_optimized.py
```

## Bonus 3

```bash
python evaluate.py --agent-class Bonus_3.train_qlearning_bonus3:Agent --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0,"smooth_mode":"confidence","smooth_dims":"angular_only","smooth_radius":1,"smooth_alpha":0.9,"confidence_tau":0.8}' --checkpoint checkpoints/base/q_learning_model.pkl --seed-base 42 --seed-count 100 --max-episode-steps 2000
```

```bash
python Bonus_3/scripts/evaluate_obs_perturb.py --agent-class Bonus_3.train_qlearning_bonus3:Agent --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0,"smooth_mode":"confidence","smooth_dims":"angular_only","smooth_radius":1,"smooth_alpha":0.9,"confidence_tau":0.8}' --checkpoint checkpoints/base/q_learning_model.pkl --noise-scale 0.05 --seed-base 42 --seed-count 100
```
