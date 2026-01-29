import os
import pickle
import time
import random
import numpy as np

import torch
import matplotlib.pyplot as plt

from intersection_env import VisionCtxIntersectionEnv
from GRU_policy import GRUGaussianPolicy
from VPG_bayesian import VPGContextFilterTrainer


def set_random_seeds(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(ctx: int, seed: int, horizon: int):
    env = VisionCtxIntersectionEnv(ctx=ctx, img_size=96, horizon=horizon, camera_range=10.0)
    env.reset(seed=seed)
    return env


def main(save_dir: str):
    # -------------------------------------------------------
    # 0. Settings
    # -------------------------------------------------------
    seed = 1337
    set_random_seeds(seed)

    tau = 0.2          # temperature (same meaning as before)
    horizon = 250      # env rollout horizon
    M = 64             # trajectories per iteration (200 may be slow for GRU + env)
    iterations = 2000  # adjust as needed
    lr = 1e-3          # learning rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------
    # 1. Environment factory (two contexts)
    # -------------------------------------------------------
    # Trainer will sample contexts internally; we pass a "base env" just for API compatibility
    base_env = make_env(ctx=1, seed=seed, horizon=horizon)

    # -------------------------------------------------------
    # 2. Policy network (GRU, history-dependent)
    # -------------------------------------------------------
    policy_net = GRUGaussianPolicy(
        obs_dim=16,      # [filled_obs(8), nan_mask(8)]
        act_dim=2,
        hidden=128,
        gru_hidden=128,
        log_std_init=-0.5,
        include_prev_action=True,
    ).to(device)

    # -------------------------------------------------------
    # 3. VPG Trainer (Context-filter based)
    # -------------------------------------------------------
    trainer = VPGContextFilterTrainer(
        env=base_env,            # used for stepping; trainer resets per-episode
        policy=policy_net,
        tau=tau,
        lr=lr,
        T=horizon,
    )

    # -------------------------------------------------------
    # 4. Logging directories
    # -------------------------------------------------------
    os.makedirs(f"{save_dir}/Values", exist_ok=True)
    os.makedirs(f"{save_dir}/Graphs", exist_ok=True)

    entropy_log = []
    value_log = []
    reward_log = []

    # -------------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------------
    for it in range(iterations):
        start_time = time.perf_counter()
        H_avg, R_avg, V_avg = trainer.train_step(M)  # dbg includes logP_y, log_post stats
        elapsed_time = time.perf_counter() - start_time

        entropy_log.append(H_avg)
        reward_log.append(R_avg)
        value_log.append(V_avg)

        print(f"Iter {it + 1}/{iterations}")
        print(f"  Avg Entropy H(C|y): {H_avg:.4f}")
        print(f"  Avg Reward:         {R_avg:.4f}")
        print(f"  Avg Value R-τH:     {V_avg:.4f}")
        print(f"Execution time: {elapsed_time:.6f} seconds")
        print("-" * 60)

    # -------------------------------------------------------
    # 6. Save policy + logs
    # -------------------------------------------------------
    torch.save(policy_net.state_dict(), f"{save_dir}/Values/policy_net_gru.pt")

    with open(f"{save_dir}/Values/entropy.pkl", "wb") as f:
        pickle.dump(entropy_log, f)
    with open(f"{save_dir}/Values/value.pkl", "wb") as f:
        pickle.dump(value_log, f)
    with open(f"{save_dir}/Values/reward.pkl", "wb") as f:
        pickle.dump(reward_log, f)

    # -------------------------------------------------------
    # 7. Plot curves
    # -------------------------------------------------------
    iters = range(iterations)

    plt.figure()
    plt.plot(iters, entropy_log)
    plt.title("Average Entropy H(C|y) per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Entropy (bits)")
    plt.savefig(f"{save_dir}/Graphs/entropy.png")
    plt.close()

    plt.figure()
    plt.plot(iters, value_log)
    plt.title("Average Value (R - τ H) per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig(f"{save_dir}/Graphs/value.png")
    plt.close()

    plt.figure()
    plt.plot(iters, reward_log)
    plt.title("Average Reward per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.savefig(f"{save_dir}/Graphs/reward.png")
    plt.close()

    print("Training complete.")


if __name__ == "__main__":
    main(save_dir="data_intersection/exp_1")
