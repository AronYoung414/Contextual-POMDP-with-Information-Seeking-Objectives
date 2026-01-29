import numpy as np
import torch

from light_dark_environment import ContinuousLightDarkPOMDP
from policy_eva import plot_trajectory, load_policy

# ===== Policy-based PG =====
from pg_pomdp_solver import RecurrentPolicy

# ===== POMCP =====
from POMCP_baseline import (
    LightDarkCPOMDP,
    POMCPSolver,
    belief_update_particles,
)


# =====================================================
# Rollout helpers
# =====================================================

def rollout_vpg(env, policy, context, x0, horizon, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    x = x0
    x_seq = [x]
    z_seq, m_seq, act_seq = [], [], []
    obs_history = []

    for t in range(horizon):
        if t == 0:
            inp = torch.zeros((1, 1, 2))
        else:
            inp = torch.tensor([obs_history], dtype=torch.float32)

        with torch.no_grad():
            probs, _ = policy(inp)

        a = np.random.choice(env.actions, p=probs[0].numpy())
        x, z, _ = env.step(x, a, context)

        x_seq.append(x)
        act_seq.append(a)

        if z is None:
            z_seq.append(0.0)
            m_seq.append(0.0)
            obs_history.append([0.0, 0.0])
        else:
            z_seq.append(float(z))
            m_seq.append(1.0)
            obs_history.append([float(z), 1.0])

    return x_seq, z_seq, m_seq, act_seq


def rollout_pg(env, policy, context, x0, horizon, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    x = x0
    x_seq = [x]
    z_seq, m_seq, act_seq = [], [], []
    obs_history = []

    for t in range(horizon):
        if t == 0:
            obs_seq = torch.zeros((1, 1, 2))
        else:
            obs_seq = torch.tensor([obs_history], dtype=torch.float32)

        with torch.no_grad():
            probs, _ = policy(obs_seq)

        probs = probs[0].numpy()
        a_idx = np.random.choice(len(env.actions), p=probs)
        a = env.actions[a_idx]

        x, z, _ = env.step(x, a, context)

        x_seq.append(x)
        act_seq.append(a)

        if z is None:
            z_seq.append(0.0)
            m_seq.append(0.0)
            obs_history.append([0.0, 0.0])
        else:
            z_seq.append(float(z))
            m_seq.append(1.0)
            obs_history.append([float(z), 1.0])

    return x_seq, z_seq, m_seq, act_seq


def rollout_pomcp(env, context, x0, horizon, seed, n_sims=300):
    np.random.seed(seed)
    env.seed(seed)
    rng = np.random.default_rng(seed)

    model = LightDarkCPOMDP()
    solver = POMCPSolver(model, n_sims=n_sims)

    particles = np.array([
        model.sample_initial_state()
        for _ in range(1000)
    ])

    x = x0
    x_seq = [x]
    z_seq, m_seq, act_seq = [], [], []

    obs = None
    last_action = None

    for t in range(horizon):
        action = solver.plan(particles, horizon - t)
        a = action.name.lower()

        x, z, _ = env.step(x, a, context)

        x_seq.append(x)
        act_seq.append(a)

        if z is None:
            z_seq.append(0.0)
            m_seq.append(0.0)
        else:
            z_seq.append(float(z))
            m_seq.append(1.0)

        if obs is not None:
            particles = belief_update_particles(
                model, particles, last_action, obs, rng
            )

        obs = z
        last_action = action

    return x_seq, z_seq, m_seq, act_seq


import matplotlib.pyplot as plt


def plot_trajectories_overlay(
        trajs,
        context,
        labels,
        colors,
        linestyles,
        light_color="#D6FFF6",
        dark_color="#F0F0F0",
        title="Trajectory Comparison",
):
    """
    trajs: list of (x_seq, z_seq, m_seq, act_seq)
    """

    T = len(trajs[0][0]) - 1
    all_x = np.concatenate([np.array(traj[0]) for traj in trajs])

    # -----------------------------
    # Light / dark background
    # -----------------------------
    if context == 0:
        light_region = (0, all_x.max() + 1)
        dark_region = (all_x.min() - 1, 0)
    else:
        light_region = (all_x.min() - 1, 0)
        dark_region = (0, all_x.max() + 1)

    plt.figure(figsize=(10, 5))

    plt.rcParams.update({
        "font.size": 16,  # base text
        "axes.labelsize": 15,  # x/y labels
        "axes.titlesize": 16,  # figure titles
        "legend.fontsize": 11,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "lines.linewidth": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    plt.axhspan(
        light_region[0],
        light_region[1],
        color=light_color,
        alpha=0.25,
        label="Light region" if context == 0 else None,
    )
    plt.axhspan(
        dark_region[0],
        dark_region[1],
        color=dark_color,
        alpha=0.25,
        label="Dark region" if context == 0 else None,
    )

    # -----------------------------
    # Plot trajectories
    # -----------------------------
    for (x_seq, z_seq, m_seq, act_seq), label, color, ls in zip(
            trajs, labels, colors, linestyles
    ):
        plt.plot(
            range(T + 1),
            x_seq,
            linestyle=ls,
            color=color,
            linewidth=2.5,
            label=label,
        )

    plt.xlabel("Time step $t$")
    plt.ylabel("Position $x$")
    plt.title(f"{title} (Context = {context})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    horizon = 20
    seed = 1337

    # ----- Environment -----
    env = ContinuousLightDarkPOMDP()

    # Fix context & initial state
    context = 0
    x0 = -1

    policy_path = "../data_ekf/exp_21_mediumVarRatio/Values/policy_net_continuous.pkl"

    # ----- Load VPG -----
    vpg_policy = load_policy(
        policy_path,
        action_dim=len(env.actions),
        hidden_dim=64,
    )

    # ----- Load policy-based PG -----
    pg_policy = RecurrentPolicy(
        obs_dim=2,
        hidden_dim=64,
        action_dim=len(env.actions),
    )
    pg_policy.load_state_dict(
        torch.load("../data_pomdp_pg/pg_policy.pkl", map_location="cpu")
    )
    pg_policy.eval()

    # ----- Rollouts -----
    vpg_traj = rollout_vpg(env, vpg_policy, context, x0, horizon, seed)
    pg_traj = rollout_pg(env, pg_policy, context, x0, horizon, seed)
    pomcp_traj = rollout_pomcp(env, context, x0, horizon, seed)

    plot_trajectories_overlay(
        trajs=[vpg_traj, pg_traj, pomcp_traj],
        context=context,
        labels=[
            "VPG (Exact Likelihood)",
            "Policy-based PG (RNN)",
            "POMCP",
        ],
        colors=["tab:blue", "tab:green", "tab:red"],
        linestyles=["-", "--", "-."],
        title="Trajectory Comparison (Same Context & Initial State)",
    )
