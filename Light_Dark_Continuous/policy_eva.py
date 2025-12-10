import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from VPG_MG import VariationalPolicyGradient
from light_dark_environment import ContinuousLightDarkPOMDP
from policy_continuous import ContinuousPolicyNetworkMasked  # <-- make sure this file exists


# ======================================================================
#  LOADING POLICY (torch checkpoint stored as .pkl)
# ======================================================================
def load_policy(path, action_dim, hidden_dim=64):
    """
    Loads a PyTorch state_dict saved using torch.save().
    Even if extension is .pkl, this is a torch checkpoint.
    """
    print(f"Loading policy from: {path}")

    state_dict = torch.load(path, map_location="cpu")

    policy = ContinuousPolicyNetworkMasked(
        action_size=action_dim,
        hidden_dim=hidden_dim,
    )

    policy.load_state_dict(state_dict)
    policy.eval()

    print("Policy successfully loaded.\n")
    return policy


# ======================================================================
#  TRAJECTORY PLOTTING
# ======================================================================
def plot_trajectory(x_seq, z_seq, m_seq, act_seq, context,
                    light_color="#D6FFF6", dark_color="#F0F0F0",
                    title="Trajectory"):
    """
    Plot continuous Light–Dark trajectory.
    """

    T = len(x_seq)

    # Determine light/dark region boundaries
    if context == 0:
        # light if x > 0
        light_region = (0, max(x_seq) + 1)
        dark_region = (min(x_seq) - 1, 0)
    else:
        # light if x < 0
        light_region = (min(x_seq) - 1, 0)
        dark_region = (0, max(x_seq) + 1)

    plt.figure(figsize=(10, 5))

    # background regions
    plt.axhspan(light_region[0], light_region[1], color=light_color, alpha=0.25)
    plt.axhspan(dark_region[0], dark_region[1], color=dark_color, alpha=0.25)

    # true state path
    plt.plot(range(T), x_seq, '-o', color='blue', label="True state x_t")

    # observations
    z_plot = [z if m == 1 else np.nan for z, m in zip(z_seq, m_seq)]
    plt.scatter(range(T), z_plot, color='red', marker='x', s=60, label="Observations z_t")

    # action text
    for t, a in enumerate(act_seq):
        plt.text(t, x_seq[t] + 0.2, a, ha='center', va='bottom', fontsize=10)

    plt.title(f"{title} (Context={context})")
    plt.xlabel("Time step t")
    plt.ylabel("Position x")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================================
#  POLICY EVALUATION
# ======================================================================
def evaluate_policy(policy_net,
                    env,
                    horizon=10,
                    num_trajs=50,
                    show_num=50,
                    seed=123,
                    save_path=None,
                    plot_one=True):
    """
    Evaluate a learned policy using exact likelihood.
    """

    vpg = VariationalPolicyGradientExactMG(env, policy_net, tau=1.0, horizon=horizon)

    np.random.seed(seed)
    env.seed(seed)

    returns, entropies, values, logliks = [], [], [], []
    traj_data = []

    for k in range(num_trajs):

        # sample full trajectory including x-sequence
        context = np.random.choice(env.contexts)
        x = env.sample_initial_state(context)

        x_seq = []
        z_seq = []
        m_seq = []
        act_seq = []
        reward = 0.0
        obs_history = []

        # ---- rollout ----
        for t in range(horizon):

            # policy input
            if t == 0:
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = policy_net(inp)
            a = np.random.choice(env.actions, p=probs[0].numpy())

            # environment step
            x, z, r = env.step(x, a, context)

            x_seq.append(x)
            reward += r

            if z is None:
                z_seq.append(0.0)
                m_seq.append(0.0)
            else:
                z_seq.append(float(z))
                m_seq.append(1.0)

            obs_history.append([z_seq[-1], m_seq[-1]])
            act_seq.append(a)

        # ---- likelihood evaluation ----
        P_y, logP_y = vpg.P_and_logP_y(z_seq, m_seq, act_seq)
        H = vpg.entropy_C_given_y(z_seq, m_seq, act_seq)

        returns.append(reward)
        entropies.append(H)
        values.append(reward - H)
        logliks.append(logP_y)

        traj_data.append({
            "context": context,
            "x_seq": x_seq,
            "z_seq": z_seq,
            "m_seq": m_seq,
            "act_seq": act_seq,
            "reward": reward,
            "logP_y": logP_y,
            "entropy": H
        })

        print(f"[{k + 1}/{num_trajs}]  R={reward:.3f}, H={H:.3f}, "
              f"V={reward - H:.3f}, logP={logP_y:.3f}")

        # optionally plot the first several trajectory
        if plot_one and k < show_num:
            plot_trajectory(x_seq, z_seq, m_seq, act_seq, context,
                            title="Sample Evaluation Trajectory")

    # ---- summary ----
    summary = {
        "avg_return": float(np.mean(returns)),
        "avg_entropy": float(np.mean(entropies)),
        "avg_value": float(np.mean(values)),
        "avg_log_likelihood": float(np.mean(logliks)),
        "num_trajs": num_trajs,
        "traj_data": traj_data
    }

    print("\n====================================")
    print("EVALUATION SUMMARY")
    print("====================================")
    print(f"Avg Return      = {summary['avg_return']:.4f}")
    print(f"Avg Entropy     = {summary['avg_entropy']:.4f}")
    print(f"Avg Value       = {summary['avg_value']:.4f}")
    print(f"Avg log P(y)    = {summary['avg_log_likelihood']:.4f}")
    print("====================================\n")

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(summary, f)
        print(f"Saved results to: {save_path}")

    return summary


# --------------------------------------------------------------
# Stand-alone Execution
# --------------------------------------------------------------
if __name__ == "__main__":
    env = ContinuousLightDarkPOMDP()

    # Change as needed
    policy_path = "data_mg/exp_3_longHorizon/Values/policy_net_continuous.pkl"

    # The observation dimension = 2 ([z, m])
    # The number of actions = len(env.actions)
    policy_net = load_policy(
        path=policy_path,
        action_dim=len(env.actions),
        hidden_dim=64,
    )

    summary = evaluate_policy(
        policy_net=policy_net,
        env=env,
        horizon=20,
        num_trajs=50,
        show_num=50,
        seed=0,
        save_path="evaluation_results.pkl",
        plot_one=True  # set to False to disable plotting
    )
