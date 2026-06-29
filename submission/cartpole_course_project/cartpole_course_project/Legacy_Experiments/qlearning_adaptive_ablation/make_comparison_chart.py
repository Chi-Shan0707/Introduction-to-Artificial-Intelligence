"""生成三种算法性能对比图（策略梯度 vs Q-Learning vs MCTS）

风格复用项目既有图表：
  - ablation_study.py: seaborn-whitegrid + bar + axhline + 数值标注
  - bonus1_analysis.py: 配色 #4C72B0 / #55A868 / #C44E52
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ── 中英混排字体 ──────────────────────────────────────────────
_CJK_FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
fp = FontProperties(fname=_CJK_FONT)
plt.rcParams["font.family"] = fp.get_name()
plt.rcParams["axes.unicode_minus"] = False
# 注册字体让 matplotlib 全局可用
from matplotlib import font_manager
font_manager.fontManager.addfont(_CJK_FONT)

# ── 数据 ──────────────────────────────────────────────────────
algos = ["策略梯度\n(Policy Gradient)", "Q-Learning", "MCTS"]
means = [2000, 593, 233]
stds  = [0,   421,  14]
colors = ["#55A868", "#4C72B0", "#C44E52"]   # 与 bonus1 boxplot 一致

# 每种算法的瓶颈说明
notes = [
    "直接拟合连续状态空间\n无需离散化",
    "受 6⁴=1296 离散\n状态分辨率限制",
    "受搜索深度与\n计算预算限制",
]
# 100% 达标标注
caps = ["100% 达到上限", "", ""]

# ── 绘图 ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6.5))

x = np.arange(len(algos))
bars = ax.bar(
    x, means, yerr=stds, width=0.55,
    color=colors, edgecolor="#333", linewidth=0.8,
    capsize=8, error_kw={"elinewidth": 1.5, "ecolor": "#555"},
    zorder=3,
)

# 上限参考线
ax.axhline(2000, color="#FF6B6B", ls="--", lw=1.2, alpha=0.7, zorder=2)
ax.text(2.42, 2000, " 上限 2000", color="#FF6B6B", fontsize=9, va="center", fontproperties=fp)

# 柱顶数值标注
for bar, m, s, cap in zip(bars, means, stds, caps):
    top = bar.get_height()
    label = f"mean={m}"
    if s > 0:
        label += f"\nstd={s}"
    if cap:
        label += f"\n{cap}"
    ax.text(
        bar.get_x() + bar.get_width() / 2, top + s + 45,
        label, ha="center", va="bottom",
        fontsize=10.5, fontweight="bold", fontproperties=fp,
    )

# 底部瓶颈说明
for i, note in enumerate(notes):
    ax.text(
        i, -180, note, ha="center", va="top",
        fontsize=8.5, color="#666", fontproperties=fp,
        linespacing=1.4,
    )

# 坐标轴
ax.set_xticks(x)
ax.set_xticklabels(algos, fontsize=11, fontproperties=fp)
ax.set_ylabel("平均存活步数 (100 seeds)", fontsize=12, fontproperties=fp)
ax.set_ylim(-320, 2650)
ax.set_xlim(-0.6, 2.7)

ax.set_title(
    "CartPole 三种算法性能对比",
    fontsize=15, fontweight="bold", color="#09397E", pad=14, fontproperties=fp,
)

# 去掉顶部/右侧脊
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "algo_comparison_styled.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
