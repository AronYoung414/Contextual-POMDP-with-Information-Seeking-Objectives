from Light_Dark_Discrete.archive_code_2.baseline_reinforce import *
import numpy as np


###############################################################
#  POLICY EVALUATION (AFTER TRAINING)
###############################################################

def evaluate_policy(env, policy, horizon=10, rollouts=50):
    """
    Evaluate the trained policy.
    Returns:
        avg_reward, avg_entropy, trajectories
    """

    all_rewards = []
    all_entropies = []
    trajectories = []

    for _ in range(rollouts):
        # sample a context
        c = random.choice(env.contexts)

        # run a trajectory under the policy
        obs_seq, act_seq, R = sample_trajectory(env, policy, horizon, c)

        # compute entropy for this (obs, act) sequence
        Hc = compute_context_entropy(env, obs_seq, act_seq)

        # record
        all_rewards.append(R)
        all_entropies.append(Hc.item())
        trajectories.append({
            "context": c,
            "observations": obs_seq,
            "actions": act_seq,
            "reward": R,
            "entropy": Hc.item()
        })

    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_entropy = sum(all_entropies) / len(all_entropies)

    return avg_reward, avg_entropy, trajectories


###############################################################
#  PRINT TRAJECTORIES IN READABLE FORMAT
###############################################################

def print_trajectories(trajectories, num_to_print=5):
    """
    Nicely prints a few sampled trajectories.
    """

    print("\n==================== SAMPLE TRAJECTORIES ====================\n")
    for i, traj in enumerate(trajectories[:num_to_print]):
        print(f"Trajectory {i + 1}:")
        print(f"  Context = {traj['context']}")
        print(f"  Observations = {traj['observations']}")
        print(f"  Actions      = {traj['actions']}")
        print(f"  Total Reward = {traj['reward']:.3f}")
        print(f"  H(C|y,a)     = {traj['entropy']:.3f}")
        print("------------------------------------------------------------\n")


###############################################################
#  SAVE TRAJECTORIES TO FILE (optional)
###############################################################

def save_trajectories(trajectories, filename="trajectories.npy"):
    np.save(filename, trajectories, allow_pickle=True)
    print(f"Saved trajectories to {filename}")


def load_policy(env, policy_path, hidden_dim=64):
    """
    Loads a trained policy from disk.
    """
    policy = PolicyNetwork(env.observations, env.actions, hidden_dim)
    state_dict = torch.load(policy_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded policy from {policy_path}")
    return policy


if __name__ == "__main__":
    set_seed(1337)

    env = FunctionalContextEnv(stoPar=0.1, obsNoise=0.1)

    ##############################
    # LOAD SAVED POLICY HERE
    ##############################
    trained_policy = load_policy(env, "../data_line_grid_3/exp_1/policy.pt", hidden_dim=64)

    # EVALUATE LOADED POLICY
    avgR, avgH, trajectories = evaluate_policy(env, trained_policy, horizon=10, rollouts=30)

    print("\n===== EVALUATION OF SAVED POLICY =====")
    print(f"Average Reward: {avgR:.3f}")
    print(f"Average Entropy H(C|y,a): {avgH:.3f}")

    print_trajectories(trajectories, num_to_print=10)
    save_trajectories(trajectories, filename="../data_line_grid_2/exp_1/trajectories_loaded_policy.npy")
