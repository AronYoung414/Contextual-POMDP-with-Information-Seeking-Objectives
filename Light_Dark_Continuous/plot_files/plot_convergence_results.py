import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load data
# -----------------------
with open("../data_ekf/exp_21_mediumVarRatio/Values/reward.pkl", "rb") as f:
    reward = pickle.load(f)

with open("../data_ekf/exp_21_mediumVarRatio/Values/entropy.pkl", "rb") as f:
    entropy = pickle.load(f)

with open("../data_ekf/exp_21_mediumVarRatio/Values/value.pkl", "rb") as f:
    values = np.array(pickle.load(f))

x_reward = range(len(reward))
x_entropy = range(len(entropy))


plt.rcParams.update({
    "font.size": 16,          # base text
    "axes.labelsize": 15,     # x/y labels
    "axes.titlesize": 16,     # figure titles
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "lines.linewidth": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =======================
# Figure 1: Reward
# =======================
# plt.figure(figsize=(3.4, 2.6))  # ICML single-column width
plt.plot(x_reward, reward, linewidth=2, color="tab:blue", label=r"total return $G_T$")
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.title("Reward")
plt.legend(loc="best", frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig("rewards.pdf", format="pdf", bbox_inches="tight")
plt.show()

# =======================
# Figure 2: Entropy
# =======================
# plt.figure(figsize=(3.4, 2.6))  # ICML single-column width
plt.plot(x_entropy, entropy, linewidth=2, color="tab:orange", label=r"entropy $H(C|Y_t)$")
plt.xlabel("Episodes")
plt.ylabel("Entropy")
plt.title("Entropy")
plt.legend(loc="best", frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig("entropy.pdf", format="pdf", bbox_inches="tight")
plt.show()

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

regret_binned = [
    regret[i*window:(i+1)*window].mean()
    for i in range(num_bins)
]

x = np.arange(1, num_bins + 1) * window

# -----------------------
# ICML-friendly plot
# -----------------------
# plt.rcParams.update({
#     "font.size": 14,
#     "axes.labelsize": 16,
#     "axes.titlesize": 16,
#     "legend.fontsize": 14,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
# })

# plt.figure(figsize=(3.4, 2.6))  # ICML single-column
plt.plot(x, regret_binned, linewidth=2, color="tab:red", label="Regret")
plt.xlabel("Episode")
plt.ylabel("Regret")
plt.title("Regret (averaged every 50 episodes)")
plt.legend(frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig("regret_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()