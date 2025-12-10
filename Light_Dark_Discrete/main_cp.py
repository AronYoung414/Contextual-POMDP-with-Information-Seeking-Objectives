import os
import pickle
import torch
import matplotlib.pyplot as plt
from line_grid_environment_cpomdp import EnvironmentCPOMDP
from variational_policy_gradient_cp import (
    PolicyNetwork,
    VariationalPolicyGradientCP,
    set_random_seeds,
)


def main(save_dir):
    # -------------------------------------------------------
    # 0. Settings
    # -------------------------------------------------------
    seed = 42
    set_random_seeds(seed)

    tau = 1.0
    horizon = 10
    M = 200          # trajectories per iteration
    iterations = 1000
    lr = 0.001

    # -------------------------------------------------------
    # 1. Create CPOMDP environment
    # -------------------------------------------------------
    env = EnvironmentCPOMDP(stoPar=0.1, obsNoise=0.1)
    env.seed(seed)

    # -------------------------------------------------------
    # 2. Policy network
    # -------------------------------------------------------
    policy_net = PolicyNetwork(
        obs_vocab_size=len(env.observations),
        action_size=env.action_size,
        hidden_dim=64,
        max_seq_len=horizon,
    )

    # -------------------------------------------------------
    # 3. Trainer
    # -------------------------------------------------------
    trainer = VariationalPolicyGradientCP(
        env=env,
        policy_net=policy_net,
        tau=tau,
        horizon=horizon,
        step_size=lr,
    )

    # -------------------------------------------------------
    # 4. Logging directories
    # -------------------------------------------------------
    os.makedirs(f"{save_dir}/Values", exist_ok=True)
    os.makedirs(f"{save_dir}/Graphs", exist_ok=True)

    entropy_log = []
    reward_log = []
    value_log = []

    # -------------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------------
    for it in range(iterations):
        H, R, V = trainer.train_step(M)

        entropy_log.append(H)
        reward_log.append(R)
        value_log.append(V)

        print(f"Iter {it+1}/{iterations}")
        print(f"  Avg Entropy: {H:.4f}")
        print(f"  Avg Reward:  {R:.4f}")
        print(f"  Avg Value:   {V:.4f}")
        print("-" * 60)

    # -------------------------------------------------------
    # Save policy
    # -------------------------------------------------------
    torch.save(policy_net.state_dict(), f"{save_dir}/Values/policy_net_cp.pkl")

    # Save logs
    with open(f"{save_dir}/Values/value.pkl", "wb") as f:
        pickle.dump(value_log, f)

    with open(f"{save_dir}/Values/entropy.pkl", "wb") as f:
        pickle.dump(entropy_log, f)

    with open(f"{save_dir}/Values/reward.pkl", "wb") as f:
        pickle.dump(reward_log, f)

    # -------------------------------------------------------
    # Plot curves
    # -------------------------------------------------------
    iters = range(iterations)

    plt.figure()
    plt.plot(iters, value_log)
    plt.title("Average Value per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig(f"{save_dir}/Graphs/value.png")
    plt.show()

    plt.figure()
    plt.plot(iters, entropy_log)
    plt.title("Entropy per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("H(C|y)")
    plt.savefig(f"{save_dir}/Graphs/entropy.png")
    plt.show()

    plt.figure()
    plt.plot(iters, reward_log)
    plt.title("Reward per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.savefig(f"{save_dir}/Graphs/reward.png")
    plt.show()

    print("Training complete.")


if __name__ == "__main__":
    main(save_dir="data_cp/exp_4_reversed")
