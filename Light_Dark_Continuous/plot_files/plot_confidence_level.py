import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# =========================
# Load data
# =========================
def load_last_2000(path):
    with open(path, "rb") as f:
        data = np.array(pickle.load(f))
    data = data.reshape(-1)
    return data[-2000:]

reward_files = [
    "data_ekf/exp_19_mediumDarkLightVar/Values/reward.pkl",
    "data_ekf/exp_21_mediumVarRatio/Values/reward.pkl",
    "data_ekf/exp_17_highDarkVar/Values/reward.pkl",
]

entropy_files = [
    "data_ekf/exp_19_mediumDarkLightVar/Values/entropy.pkl",
    "data_ekf/exp_21_mediumVarRatio/Values/entropy.pkl",
    "data_ekf/exp_17_highDarkVar/Values/entropy.pkl",
]

# reward_files = [
#     "data_ekf/exp_18_highDarkLightVar/Values//reward.pkl",
#     "data_ekf/exp_19_mediumDarkLightVar/Values/reward.pkl",
#     "data_ekf/exp_20_lowDarkLightVar/Values/reward.pkl",
# ]
#
# entropy_files = [
#     "data_ekf/exp_18_highDarkLightVar/Values/entropy.pkl",
#     "data_ekf/exp_19_mediumDarkLightVar/Values/entropy.pkl",
#     "data_ekf/exp_20_lowDarkLightVar/Values/entropy.pkl",
# ]

entropy_groups = [load_last_2000(p) for p in entropy_files]
reward_groups = [load_last_2000(p) for p in reward_files]

labels = ["low", "Medium", "High"]

# =========================
# Load data
# =========================
entropy_groups = [load_last_2000(p) for p in entropy_files]
reward_groups  = [load_last_2000(p) for p in reward_files]

# =========================
# Median + 95% quantiles
# =========================
def median_quantiles(data):
    med = np.median(data)
    q_low = np.quantile(data, 0.025)
    q_high = np.quantile(data, 0.975)
    return med, q_low, q_high

entropy_stats = [median_quantiles(g) for g in entropy_groups]
reward_stats  = [median_quantiles(g) for g in reward_groups]

entropy_med  = np.array([s[0] for s in entropy_stats])
entropy_low  = np.array([s[1] for s in entropy_stats])
entropy_high = np.array([s[2] for s in entropy_stats])

reward_med  = np.array([s[0] for s in reward_stats])
reward_low  = np.array([s[1] for s in reward_stats])
reward_high = np.array([s[2] for s in reward_stats])

x = np.arange(len(labels))

# =========================
# Colors (pretty & ICML-safe)
# =========================
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green
markers = ["o", "s", "^"]

# =========================
# ICML-friendly style
# =========================
FIG_WIDTH = 8.5
FIG_HEIGHT = 4.5

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 15,
    "lines.linewidth": 2.5,
})

fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

# =========================
# Entropy plot
# =========================
for i in range(len(labels)):
    axes[0].errorbar(
        x[i],
        entropy_med[i],
        yerr=[[entropy_med[i] - entropy_low[i]],
              [entropy_high[i] - entropy_med[i]]],
        fmt=markers[i],
        color=colors[i],
        ecolor=colors[i],
        capsize=8,
        markersize=10,
        elinewidth=2,
        label=labels[i]
    )

# Connect medians (neutral color)
axes[0].plot(x, entropy_med, color="black", linewidth=2, alpha=0.6)

axes[0].set_title("Entropy")
axes[0].set_ylabel("Values")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].grid(alpha=0.3)
# axes[0].legend(frameon=False)

# =========================
# Reward plot
# =========================
for i in range(len(labels)):
    axes[1].errorbar(
        x[i],
        reward_med[i],
        yerr=[[reward_med[i] - reward_low[i]],
              [reward_high[i] - reward_med[i]]],
        fmt=markers[i],
        color=colors[i],
        ecolor=colors[i],
        capsize=8,
        markersize=10,
        elinewidth=2,
        label=labels[i]
    )

# Connect medians (neutral color)
axes[1].plot(x, reward_med, color="black", linewidth=2, alpha=0.6)

axes[1].set_title("Return")
axes[0].set_ylabel("Values")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].grid(alpha=0.3)
# axes[1].legend(frameon=False)

# =========================
# Layout + save
# =========================
plt.tight_layout()
plt.savefig("median_quantile_colored_errorbars.pdf", bbox_inches="tight")
plt.show()