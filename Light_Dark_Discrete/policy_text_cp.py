# ---------------------------------------------------------
# Normal script version (NOT pytest)
# Run this with:
#      python policy_test_cp.py
# ---------------------------------------------------------

import torch
from line_grid_environment_cpomdp import EnvironmentCPOMDP
from variational_policy_gradient_cp import (
    PolicyNetwork,
    VariationalPolicyGradientCP,
    set_random_seeds,
)


def test_policy_cp(policy_path="data_cp/Values/policy_net_cp.pkl",
                   M=50, T=10, seed_val=1337):
    """
    Generate and print trajectories using a trained policy.
    This is a normal script function and NOT a pytest test.
    """

    # -------------------------------------------------------
    # 0. Setup
    # -------------------------------------------------------
    set_random_seeds(seed_val)

    # -------------------------------------------------------
    # 1. Build CPOMDP environment
    # -------------------------------------------------------
    env = EnvironmentCPOMDP(stoPar=0.1, obsNoise=0.1)
    env.seed(seed_val)

    # -------------------------------------------------------
    # 2. Build policy network and load weights
    # -------------------------------------------------------
    policy_net = PolicyNetwork(
        obs_vocab_size=len(env.observations),
        action_size=env.action_size,
        hidden_dim=64,
        max_seq_len=T,
    )

    # load policy
    print(f"Loading policy from: {policy_path}")
    state_dict = torch.load(policy_path, map_location="cpu")
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    # -------------------------------------------------------
    # 3. Wrap trainer (only for sampling, not training)
    # -------------------------------------------------------
    trainer = VariationalPolicyGradientCP(
        env=env,
        policy_net=policy_net,
        tau=1.0,
        horizon=T,
        step_size=0.001,
    )

    # -------------------------------------------------------
    # 4. Generate M trajectories
    # -------------------------------------------------------
    print(f"\nGenerating {M} trajectories...\n")

    for m in range(M):
        context, st_list, obs_list, act_list, reward = \
            trainer.sample_one_trajectory()

        # compute entropy
        H, _ = trainer.compute_entropy_and_logp(obs_list, act_list)

        print(f"Trajectory {m}")
        print("  context:     ", context)
        print("  states:      ", st_list)
        print("  actions:     ", act_list)
        print("  observations:", obs_list)
        print("  total reward:", reward)
        print("  H(C | y):    ", H.item())
        print("-" * 70)


# ---------------------------------------------------------
# Run directly as script
# ---------------------------------------------------------
if __name__ == "__main__":
    # Path to your saved policy
    policy_path = "data_cp/exp_4_reversed/Values/policy_net_cp.pkl"

    # Run test
    test_policy_cp(policy_path)
