import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("../data_ekf/exp_21_mediumVarRatio/Values/value.pkl", "rb") as f:
    values = np.array(pickle.load(f))

# -----------------------
# Compute regret
# -----------------------
V_star = values.max()
regret = V_star - values

# -----------------------
# Aggregate every 50 episodes
# -----------------------
window = 50
num_bins = len(regret) // window

regret_binned_our = [
    regret[i*window:(i+1)*window].mean()
    for i in range(num_bins)
]

x = np.arange(1, num_bins + 1) * window

# =====================
# Load data
# =====================

with open("../data_pomcp/pomcp_cached_policy.pkl", "rb") as f:
    data = pickle.load(f)


returns = np.array(data["returns"])


# =====================
# Regret definition
# =====================
# V_star = returns.max()          # empirical best
regret = V_star - returns       # per-episode regret

window = 50
num_bins = len(regret) // window

regret_binned_mct = [
    regret[i*window:(i+1)*window].mean()
    for i in range(num_bins)
]

with open("../data_pomdp_pg/pg_train_returns.pkl", "rb") as f:
    data = pickle.load(f)

# add more arrays here when you have them (pomcp_returns, cached_pomcp_returns, vpg_returns, etc.)
pg = np.array(data["train_returns"], dtype=np.float32)

# V_star = float(pg.max())  # if only one method present now; later use max over all methods

regret_pg = V_star - pg

regret_binned_pg = [
    regret_pg[i*window:(i+1)*window].mean()
    for i in range(num_bins)
]

plt.rcParams.update({
    "font.size": 16,          # base text
    "axes.labelsize": 15,     # x/y labels
    "axes.titlesize": 16,     # figure titles
    "legend.fontsize": 11,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# plt.figure(figsize=(3.4, 2.6))  # ICML single-column
plt.plot(x, regret_binned_our,  linewidth=2, color="tab:red", label="C-IDS")
plt.plot(x, regret_binned_mct, linestyle='dashed', linewidth=2, label="POMCP")
plt.plot(x, regret_binned_pg, linestyle= 'dotted',  linewidth=2, label="PG-POMDP")
plt.xlabel("Episode")
plt.ylabel("Regret")
plt.title("Regret (averaged every 50 episodes)")
plt.legend(frameon=False, loc=4, bbox_to_anchor=(1, 0.06))
plt.grid(True)
plt.tight_layout()
plt.savefig("regret_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()
