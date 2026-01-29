import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================
# ICML-friendly font setup
# ==========================
plt.rcParams.update({
    "font.size": 12,          # base
    "axes.titlesize": 14,     # subplot titles
    "axes.labelsize": 13,     # axis labels
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ==========================
# Environment parameters
# ==========================
light_noise = 1.0
dark_noise = 8.0

z_range = np.linspace(-6, 6, 1200)
x_light = 3.0
x_dark = -3.0

# ==========================
# Helper functions
# ==========================
def plot_obs(ax, mu, var, label, color):
    sigma = np.sqrt(var)
    pdf = norm.pdf(z_range, mu, sigma)
    ax.plot(z_range, pdf, linewidth=2.5, color=color, label=label)

def shade_region(ax, x_min, x_max, color, alpha, label=None):
    ax.axvspan(x_min, x_max, color=color, alpha=alpha, label=label)

# ==========================
# Figure: SIDE BY SIDE
# ==========================
fig, axes = plt.subplots(
    1, 2,
    figsize=(6.6, 3.2),   # ~2x ICML column width
    sharey=True
)

# ======================================================
# Context 0
# ======================================================
ax = axes[0]

shade_region(ax, -6, 0, color="black", alpha=0.20, label="Dark")
shade_region(ax, 0, 6, color="lightgray", alpha=0.45, label="Light")

plot_obs(ax, x_light, light_noise, "Light obs", "tab:blue")
plot_obs(ax, x_dark, dark_noise, "Dark obs", "tab:red")

# Reward / penalty markers
ax.axvline(1, linestyle="--", linewidth=2, color="black")
ax.text(1.05, ax.get_ylim()[1]*0.85, "Reward",
        rotation=90, va="top", fontsize=9)

ax.axvline(0, linestyle="--", linewidth=1.8, color="black")
ax.text(-0.05, ax.get_ylim()[1]*0.85, "Penalty",
        rotation=90, va="top", ha="right", fontsize=9)

ax.set_title("Context 0\nLight: $x>0$ | Dark: $x \leq 0$")
ax.set_xlabel("State")
ax.set_ylabel("Obs. density")
ax.set_xlim(-6, 6)
ax.grid(alpha=0.3)
ax.legend(frameon=False)

# ======================================================
# Context 1
# ======================================================
ax = axes[1]

shade_region(ax, -6, 0, color="lightgray", alpha=0.45, label="Light")
shade_region(ax, 0, 6, color="black", alpha=0.20, label="Dark")

plot_obs(ax, x_dark, light_noise, "Light obs", "tab:blue")
plot_obs(ax, x_light, dark_noise, "Dark obs", "tab:red")

ax.axvline(-1, linestyle="--", linewidth=2, color="black")
ax.text(-1.05, ax.get_ylim()[1]*0.85, "Reward",
        rotation=90, va="top", ha="right", fontsize=9)

ax.axvline(0, linestyle="--", linewidth=1.8, color="black")
ax.text(0.05, ax.get_ylim()[1]*0.85, "Penalty",
        rotation=90, va="top", ha="left", fontsize=9)

ax.set_title("Context 1\nLight: $x<0$ | Dark: $x \geq 0$")
ax.set_xlabel("State")
ax.set_xlim(-6, 6)
ax.grid(alpha=0.3)
ax.legend(frameon=False)

# ==========================
# Final layout
# ==========================
plt.tight_layout(pad=0.3, w_pad=0.6)
plt.savefig("environment_side_by_side.pdf", bbox_inches="tight")
plt.show()
