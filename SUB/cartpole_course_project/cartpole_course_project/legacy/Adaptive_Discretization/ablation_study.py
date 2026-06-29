
"""Ablation: adapt one dimension at a time. 3 runs per config, pick best."""
import os, sys, pickle, json
from datetime import datetime
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = 0.04; GAMMA = 0.99; EPSILON = 0.1; EPOCHS = 2000
TARGET = 1000; MAX_DIG = 6; ADAPT_INTERVAL = 250; GRAD_ALPHA = 15.0
CLIP_RANGES = [(-2.4, 2.4), (-3.0, 3.0), (-0.5, 0.5), (-2.0, 2.0)]
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "legacy_adaptive"))
N_RUNS = 3  # run each config 3 times, pick best

class Agent:
    def __init__(self, n_state, n_action, adapt_dims=None, lr=LR, gamma=GAMMA, epsilon=EPSILON):
        self.n_action = n_action; self.n_state = n_state
        self.lr = lr; self.gamma = gamma; self.epsilon = epsilon
        self.adapt_dims = adapt_dims
        self.bin_boundaries = [np.linspace(lo, hi, MAX_DIG+1) for lo, hi in CLIP_RANGES]
        self.q_table = np.random.uniform(0, 1, (MAX_DIG**n_state, n_action))
        self.td_errors = np.zeros(MAX_DIG**n_state)
        self.visit_counts = np.zeros(MAX_DIG**n_state)

    def _s2b(self, sid):
        return [sid%MAX_DIG, (sid//MAX_DIG)%MAX_DIG, (sid//MAX_DIG**2)%MAX_DIG, (sid//MAX_DIG**3)%MAX_DIG]
    def _b2s(self, bs):
        return bs[0] + bs[1]*MAX_DIG + bs[2]*MAX_DIG**2 + bs[3]*MAX_DIG**3

    def _dim_mask(self, dim, bi):
        m = np.zeros(MAX_DIG**4, dtype=bool)
        for sid in range(MAX_DIG**4):
            if self._s2b(sid)[dim] == bi: m[sid] = True
        return m

    def make_bins_state(self, state):
        s = []
        for val, bd in zip(state, self.bin_boundaries):
            val = np.clip(val, bd[0], bd[-1])
            idx = np.searchsorted(bd[1:-1], val, side="right")
            s.append(min(idx, MAX_DIG-1))
        return self._b2s(s)

    def _bin_center(self, sid, bds):
        bs = self._s2b(sid)
        return [(bds[i][bs[i]] + bds[i][bs[i]+1])/2.0 for i in range(4)]

    def _lookup_with_bds(self, state, bds):
        s = []
        for val, bd in zip(state, bds):
            val = np.clip(val, bd[0], bd[-1])
            idx = np.searchsorted(bd[1:-1], val, side="right")
            s.append(min(idx, MAX_DIG-1))
        return self._b2s(s)

    def _remap(self, old_bds, momentum=0.3):
        old_q = self.q_table.copy()
        new_q = np.zeros_like(self.q_table)
        for sid in range(MAX_DIG**4):
            c = self._bin_center(sid, self.bin_boundaries)
            old_sid = self._lookup_with_bds(c, old_bds)
            new_q[sid] = old_q[old_sid]
        self.q_table = momentum*self.q_table + (1-momentum)*new_q

    def _adapt(self):
        old_bds = [b.copy() for b in self.bin_boundaries]
        dims_to_adapt = self.adapt_dims if self.adapt_dims is not None else range(4)
        for dim in dims_to_adapt:
            lo, hi = CLIP_RANGES[dim]
            bds = self.bin_boundaries[dim]
            qpb = []
            for b in range(MAX_DIG):
                mask = self._dim_mask(dim, b)
                if np.any(mask):
                    qpb.append(float(np.mean(np.max(self.q_table[mask,:], axis=1))))
                else:
                    qpb.append(0.0)
            grads = [abs(qpb[b+1]-qpb[b])+0.005 for b in range(MAX_DIG-1)]
            cd = [0.0]
            for b in range(MAX_DIG):
                w = bds[b+1]-bds[b]
                d = 1.0 + GRAD_ALPHA*(grads[b] if b < len(grads) else grads[-1])
                cd.append(cd[-1]+d*w)
            td = cd[-1]
            if td < 0.001: continue
            nb = [lo]
            for i in range(1, MAX_DIG):
                tgt = td*i/MAX_DIG
                for j in range(len(cd)-1):
                    if cd[j] <= tgt <= cd[j+1]:
                        fr = (tgt-cd[j])/(cd[j+1]-cd[j]+1e-8)
                        nb.append(float(bds[j]+fr*(bds[j+1]-bds[j])))
                        break
            nb.append(hi)
            self.bin_boundaries[dim] = np.array(nb)
        self._remap(old_bds)

    def update_q_table(self, s, a, r, ns):
        ci = self.make_bins_state(s); ni = self.make_bins_state(ns)
        tgt = r + self.gamma*np.max(self.q_table[ni,:])
        td = tgt - self.q_table[ci,a]
        self.q_table[ci,a] += self.lr*td
        self.td_errors[ci] = 0.9*self.td_errors[ci] + 0.1*abs(td)

    def decide_action(self, s):
        ci = self.make_bins_state(s)
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(self.n_action)
        return int(np.argmax(self.q_table[ci,:]))

    def predict(self, s):
        return int(np.argmax(self.q_table[self.make_bins_state(s),:]))

    def save_model(self, fp):
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        with open(fp,"wb") as f:
            pickle.dump({"q_table":self.q_table,"bin_boundaries":self.bin_boundaries,
                "adapt_dims":self.adapt_dims,"lr":self.lr,"gamma":self.gamma,
                "epsilon":self.epsilon,"n_state":self.n_state,"n_action":self.n_action,
                "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)

    def load_model(self, fp):
        with open(fp,"rb") as f: d = pickle.load(f)
        self.q_table = d["q_table"]; self.bin_boundaries = d["bin_boundaries"]
        self.lr=d["lr"]; self.gamma=d["gamma"]; self.epsilon=d["epsilon"]
        self.n_state=d["n_state"]; self.n_action=d["n_action"]


def train_once(adapt_dims, seed=0):
    np.random.seed(seed)
    env = gym.make("CartPole-v1", max_episode_steps=TARGET)
    ns = env.observation_space.shape[0]; na = env.action_space.n
    agent = Agent(ns, na, adapt_dims=adapt_dims)
    record = []
    for i in range(EPOCHS):
        if i > 0 and i % ADAPT_INTERVAL == 0:
            if adapt_dims is None or len(adapt_dims) > 0:
                agent._adapt()
        s = env.reset()[0]
        for step in range(TARGET):
            a = agent.decide_action(s)
            ns_, _, term, trunc, _ = env.step(a)
            r = (step-TARGET) if (term or trunc) else 0
            agent.update_q_table(s, a, r, ns_)
            s = ns_
            if term or trunc:
                record.append(step)
                break
    env.close()
    return record, agent


def eval_agent(agent, perturb=0.0, n_seeds=100):
    agent.epsilon = 0.0
    steps_list = []
    for seed in range(42, 42+n_seeds):
        env = gym.make("CartPole-v1", max_episode_steps=2000)
        st, _ = env.reset(seed=seed)
        st = np.array(st, dtype=np.float32)
        if perturb > 0:
            rng = np.random.default_rng(seed)
            st = st + rng.normal(0, perturb, size=st.shape).astype(np.float32)
            env.unwrapped.state = st
        steps = 0
        while True:
            a = int(agent.predict(st))
            st, _, term, trunc, _ = env.step(a)
            steps += 1
            if term or trunc: break
        env.close()
        steps_list.append(steps)
    return float(np.mean(steps_list)), float(np.median(steps_list)), float(np.mean(np.array(steps_list)>=2000))


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    experiments = [
        (None, "all_dims"),
        (set(), "no_adapt"),
        ({0}, "dim0_pos"),
        ({1}, "dim1_vel"),
        ({2}, "dim2_angle"),
        ({3}, "dim3_angvel"),
    ]

    results = {}
    for adapt_dims, tag in experiments:
        print(f"\n{'='*60}")
        print(f"Config: {tag} (adapt_dims={adapt_dims}), running {N_RUNS}x")
        print(f"{'='*60}")
        
        best_mean = -1; best_agent = None; best_record = None
        for run in range(N_RUNS):
            record, agent = train_once(adapt_dims, seed=run*42)
            last20 = np.mean(record[-20:])
            print(f"  Run {run}: last20={last20:.0f}")
            if last20 > best_mean:
                best_mean = last20; best_agent = agent; best_record = record
        
        fp = os.path.join(SAVE_DIR, f"ablation_{tag}.pkl")
        best_agent.save_model(fp)
        print(f"  Best last20={best_mean:.0f}, saved to {fp}")

        for perturb in [0.0, 0.05]:
            mean_s, med_s, ratio = eval_agent(best_agent, perturb=perturb)
            key = f"{tag}_p{perturb}"
            results[key] = {"tag": tag, "perturb": perturb, "mean": mean_s, "median": med_s, "ratio_2000": ratio}
            print(f"  Eval p={perturb}: mean={mean_s:.1f}, med={med_s:.0f}, ratio={ratio*100:.0f}%")

    with open(os.path.join(SAVE_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Chart
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    tags_order = ["no_adapt", "dim0_pos", "dim1_vel", "dim2_angle", "dim3_angvel", "all_dims"]
    labels = ["Baseline\n(no adapt)", "dim0 only\n(cart_pos)", "dim1 only\n(cart_vel)",
              "dim2 only\n(pole_angle)", "dim3 only\n(pole_angvel)", "All dims\n(full)"]
    means_p0 = [results[f"{t}_p0.0"]["mean"] for t in tags_order]
    means_p05 = [results[f"{t}_p0.05"]["mean"] for t in tags_order]
    colors = ["#A0C4E8","#C8D8E8","#C8D8E8","#C8D8E8","#7EB5D6","#1E6FBA"]

    for ax, means, title in [(ax1, means_p0, "perturb = 0"), (ax2, means_p05, "perturb = 0.05")]:
        bars = ax.bar(range(len(tags_order)), means, color=colors, edgecolor="#333", lw=0.8)
        ax.set_xticks(range(len(tags_order))); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Mean Steps (100 seeds)", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", color="#09397E")
        ax.axhline(2000, color="#FF6B6B", ls="--", lw=1, alpha=0.7)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, f"{val:.0f}",
                    ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ablation_chart.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved.")

    print(f"\n{'='*70}")
    print(f"{'Method':<20} {'Mean(p=0)':>12} {'Mean(p=.05)':>12} {'Ratio(p=0)':>12}")
    print(f"{'-'*70}")
    for t in tags_order:
        r0 = results[f"{t}_p0.0"]; r05 = results[f"{t}_p0.05"]
        print(f"{t:<20} {r0['mean']:>12.1f} {r05['mean']:>12.1f} {r0['ratio_2000']*100:>11.0f}%")
    print(f"{'='*70}")
