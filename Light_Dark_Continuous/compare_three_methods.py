import numpy as np
import torch
import pickle
from scipy.stats import t

from collections import Counter

# ===== Your environment =====
from light_dark_environment import ContinuousLightDarkPOMDP

# ===== Your VPG evaluator =====
from policy_eva import evaluate_policy, load_policy

# ===== Policy-based POMDP =====
from pg_pomdp_solver import RecurrentPolicy, sample_context

# ===== POMCP =====
from POMCP_baseline import LightDarkCPOMDP, POMCPSolver
from POMCP_baseline import belief_update_particles

# =====================================================
# Utilities
# =====================================================
def confidence_interval(x, alpha=0.05):
    """
    95% CI using Student-t
    """
    x = np.asarray(x)
    n = len(x)
    mean = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    h = sem * t.ppf(1 - alpha / 2, n - 1)
    return mean, h


# =====================================================
# Evaluate policy-based PG
# =====================================================
def eval_policy_pg(env, policy, horizon, seeds):
    rewards = []
    entropies = []

    policy.eval()

    for sd in seeds:
        np.random.seed(sd)
        torch.manual_seed(sd)
        env.seed(sd)

        context = sample_context(env, np.random.default_rng(sd))
        x = env.sample_initial_state(context)

        obs_hist = []
        total_reward = 0.0
        entropy_t = []

        for t in range(horizon):
            if t == 0:
                obs_seq = torch.zeros((1, 1, 2))
            else:
                obs_seq = torch.tensor([obs_hist], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = policy(obs_seq)

            probs_np = probs[0].numpy()
            entropy_t.append(-np.sum(probs_np * np.log(probs_np + 1e-12)))

            a_idx = np.random.choice(len(env.actions), p=probs_np)
            a = env.actions[a_idx]

            x, z, r = env.step(x, a, context)
            total_reward += r

            if z is None:
                obs_hist.append([0.0, 0.0])
            else:
                obs_hist.append([float(z), 1.0])

        rewards.append(total_reward)
        entropies.append(np.mean(entropy_t))

    return np.array(rewards), np.array(entropies)


# =====================================================
# Evaluate POMCP (online, reduced budget)
# =====================================================
def eval_pomcp(env, horizon, seeds, n_sims=300):
    rewards = []
    entropies = []

    for sd in seeds:
        env.seed(sd)
        rng = np.random.default_rng(sd)

        context = rng.choice(
            env.contexts,
            p=[env.context_distribution[c] for c in env.contexts]
        )
        x = env.sample_initial_state(context)

        model = LightDarkCPOMDP()
        solver = POMCPSolver(model, n_sims=n_sims)

        particles = np.array([
            model.sample_initial_state()
            for _ in range(1000)
        ])

        total_reward = 0.0
        action_counter = Counter()
        obs = None
        last_action = None

        for t in range(horizon):
            action = solver.plan(particles, horizon - t)
            action_counter[action.name] += 1

            x, z, r = env.step(x, action.name.lower(), context)
            total_reward += r

            if obs is not None:
                particles = belief_update_particles(
                    model, particles, last_action, obs, rng
                )

            obs = z
            last_action = action

        rewards.append(total_reward)

        counts = np.array([action_counter[a] for a in ["L", "R", "O"]], dtype=float)
        p = counts / (counts.sum() + 1e-12)
        entropies.append(-np.sum(p * np.log(p + 1e-12)))

    return np.array(rewards), np.array(entropies)


# =====================================================
# MAIN COMPARISON
# =====================================================
if __name__ == "__main__":
    env = ContinuousLightDarkPOMDP()
    horizon = 20
    num_trajs = 200
    seeds = list(range(1000, 1000 + num_trajs))

    # ---------------- VPG ----------------
    # Change as needed
    policy_path = "data_ekf/exp_21_mediumVarRatio/Values/policy_net_continuous.pkl"

    # The observation dimension = 2 ([z, m])
    # The number of actions = len(env.actions)
    vpg_policy = load_policy(
        path=policy_path,
        action_dim=len(env.actions),
        hidden_dim=64,
    )

    vpg_summary = evaluate_policy(
        vpg_policy,
        env,
        horizon=horizon,
        num_trajs=num_trajs,
        plot_one=False,
    )

    vpg_rewards = np.array([d["reward"] for d in vpg_summary["traj_data"]])
    vpg_entropy = np.array([d["entropy"] for d in vpg_summary["traj_data"]])

    # ---------------- Policy-based PG ----------------
    pg_policy = RecurrentPolicy(obs_dim=2, hidden_dim=64, action_dim=len(env.actions))
    pg_policy.load_state_dict(torch.load("data_pomdp_pg/pg_policy.pkl", map_location="cpu"))

    pg_rewards, pg_entropy = eval_policy_pg(env, pg_policy, horizon, seeds)

    # ---------------- POMCP ----------------
    pomcp_rewards, pomcp_entropy = eval_pomcp(env, horizon, seeds)

    # =====================================================
    # Report
    # =====================================================
    def report(name, rewards, entropy):
        r_mean, r_ci = confidence_interval(rewards)
        h_mean, h_ci = confidence_interval(entropy)

        print(f"\n{name}")
        print("-" * len(name))
        print(f"Reward  : {r_mean:.3f} ± {r_ci:.3f}")
        print(f"Entropy : {h_mean:.3f} ± {h_ci:.3f}")

    report("VPG (Exact Likelihood)", vpg_rewards, vpg_entropy)
    report("Policy-based PG (RNN)", pg_rewards, pg_entropy)
    report("POMCP (Online)", pomcp_rewards, pomcp_entropy)
