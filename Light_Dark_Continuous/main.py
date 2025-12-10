import os
import pickle
import time
import numpy as np, random

import torch
import matplotlib.pyplot as plt

from light_dark_environment import ContinuousLightDarkPOMDP
from policy_continuous import ContinuousPolicyNetworkMasked
from VPG_EKF import VariationalPolicyGradientExact


def set_random_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(save_dir):
    # -------------------------------------------------------
    # 0. Settings
    # -------------------------------------------------------
    seed = 42
    set_random_seeds(seed)

    tau = 1  # temperature
    horizon = 10  # T
    M = 200  # trajectories per iteration
    iterations = 1000
    lr = 0.001  # learning rate
    # num_particles = 2000  # PF particles

    # -------------------------------------------------------
    # 1. Continuous Light-Dark POMDP
    # -------------------------------------------------------
    env = ContinuousLightDarkPOMDP()
    env.seed(seed)

    # -------------------------------------------------------
    # 2. Policy network (continuous, masked)
    # -------------------------------------------------------
    policy_net = ContinuousPolicyNetworkMasked(action_size=len(env.actions),
                                               hidden_dim=64)

    # -------------------------------------------------------
    # 3. Particle Filter
    # -------------------------------------------------------
    # pf = ParticleFilter(env, policy_net, num_particles=num_particles)

    # -------------------------------------------------------
    # 4. VPG Trainer (PF-based)
    # -------------------------------------------------------
    # trainer = VariationalPolicyGradientContinuousPF(
    #     env=env,
    #     policy_net=policy_net,
    #     pf=pf,
    #     tau=tau,
    #     horizon=horizon,
    #     lr=lr,
    # )
    # -------------------------------------------------------
    # 4. VPG Trainer (KF-based)
    # -------------------------------------------------------
    trainer = VariationalPolicyGradientExact(
        env=env,
        policy_net=policy_net,
        tau=tau,
        horizon=horizon,
        lr=lr,
    )
    # -------------------------------------------------------
    # 5. Logging directories
    # -------------------------------------------------------
    os.makedirs(f"{save_dir}/Values", exist_ok=True)
    os.makedirs(f"{save_dir}/Graphs", exist_ok=True)

    entropy_log = []
    value_log = []
    reward_log = []
    # (entropy removed because we no longer have discrete H(C|y);
    #  value already includes entropy via logP)

    # -------------------------------------------------------
    # 6. Training loop
    # -------------------------------------------------------
    for it in range(iterations):
        start_time = time.perf_counter()
        H_avg, R_avg, V_avg = trainer.train_step(M)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        entropy_log.append(H_avg)
        reward_log.append(R_avg)
        value_log.append(V_avg)

        print(f"Iter {it + 1}/{iterations}")
        print(f"  Avg Entropy: {H_avg:.4f}")
        print(f"  Avg Reward: {R_avg:.4f}")
        print(f"  Avg Value:  {V_avg:.4f}")
        print(f"Execution time: {elapsed_time:.6f} seconds")
        print("-" * 60)

    # -------------------------------------------------------
    # 7. Save policy
    # -------------------------------------------------------
    torch.save(policy_net.state_dict(), f"{save_dir}/Values/policy_net_continuous.pkl")

    # Save logs
    with open(f"{save_dir}/Values/entropy.pkl", "wb") as f:
        pickle.dump(entropy_log, f)

    with open(f"{save_dir}/Values/value.pkl", "wb") as f:
        pickle.dump(value_log, f)

    with open(f"{save_dir}/Values/reward.pkl", "wb") as f:
        pickle.dump(reward_log, f)

    # -------------------------------------------------------
    # 8. Plot curves
    # -------------------------------------------------------
    iters = range(iterations)

    plt.figure()
    plt.plot(iters, entropy_log)
    plt.title("Average Entropy per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Entropy")
    plt.savefig(f"{save_dir}/Graphs/entropy.png")
    plt.show()

    plt.figure()
    plt.plot(iters, value_log)
    plt.title("Average Value per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig(f"{save_dir}/Graphs/value.png")
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
    main(save_dir="data_ekf/exp_4")
