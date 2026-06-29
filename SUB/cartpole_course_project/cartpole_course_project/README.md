# CartPole Course Project

## 目录

```text
cartpole_course_project/
├── train_qlearning.py
├── train_reinforce.py
├── train_mcts.py
├── evaluate.py
├── vis.py
├── vis_mcts.py
├── agents.md
├── checkpoints/
│   ├── base/
│   ├── bonus2/
│   ├── bonus3/
│   └── legacy_adaptive/
├── Bonus_1/
├── Bonus_2/
├── Bonus_3/
└── legacy/Adaptive_Discretization/
```

## Checkpoint

- `checkpoints/base/`：基础 Q-learning、REINFORCE
- `checkpoints/bonus2/`：Bonus 2 优化 Q-learning
- `checkpoints/bonus3/`：Bonus 3 观测噪声训练模型
- `checkpoints/legacy_adaptive/`：自适应离散化消融
- MCTS 无 checkpoint

## 运行

```bash
pip install -r requirements.txt
```

基础 Q-learning：

```bash
python evaluate.py --agent-class train_qlearning:Agent --agent-init-kwargs '{"n_state":4,"n_action":2,"lr":0.04,"gamma":0.99,"epsilon":0.0}' --checkpoint checkpoints/base/q_learning_model.pkl --seed-base 42 --seed-count 100 --max-episode-steps 2000
```

基础 REINFORCE：

```bash
python evaluate.py --agent-class train_reinforce:Agent --agent-init-kwargs '{"n_state":4,"hidden_c":16,"n_action":2}' --checkpoint checkpoints/base/reinforce_model.pt --seed-base 42 --seed-count 100 --max-episode-steps 2000
```

MCTS：

```bash
python Bonus_1/evaluate_mcts.py --seed-base 42 --seed-count 100 --iteration-budget 80 --lookahead-target 200 --start-cp 200 --max-episode-steps 2000
```

Bonus 命令见 `agents.md`。
