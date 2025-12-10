from variational_policy_gradient import *
import torch


def generate_trajectories_from_saved_policy(policy_path, M=50, T=10, seed_val=1337):
    """
    Generate trajectories using a pretrained policy.
    Fixes:
      - No random initial context at construction
      - Reset observable-operator cache after each context switch
      - Correct environment seeding
      - Ensures consistent initial observation sequence
      - Makes sampling behavior identical to training
    """

    # ---------------------------
    # 1. Set seed
    # ---------------------------
    set_random_seeds(seed_val)

    # ---------------------------
    # 2. Rebuild environment EXACTLY as in training
    # ---------------------------
    # IMPORTANT: No random choice([0,1]) here
    # Context will be switched per trajectory inside sample_data
    exp_name = 'line_grid_1'
    env = Environment(context=0, stoPar=0.1, obsNoise=0.1)
    env.seed(seed_val)

    # ---------------------------
    # 3. Rebuild the policy network (same architecture)
    # ---------------------------
    policy_net = PolicyNetwork(
        obs_vocab_size=len(env.observations),
        action_size=env.action_size,
        hidden_dim=64,
        max_seq_len=T,
    )

    state_dict = torch.load(policy_path, map_location="cpu")
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    # ---------------------------
    # 4. Build a VariationalPolicyGradient instance
    #    ONLY to reuse sample_data() and operator tools.
    # ---------------------------
    trainer = VariationalPolicyGradient(
        exp_name=exp_name,
        env=env,
        experiment_num=999,  # irrelevant
        tau=1.0,
        gamma=1,
        horizon=T,
        trajectories_num=M,
        iteration_num=0,
        step_size=0.001,
        policy_network=policy_net
    )

    # ---------------------------
    # CRITICAL FIX:
    # Observable operators must be cleared after context change
    # ---------------------------
    def safe_set_context(c):
        trainer.env.set_context(c)
        # Reset operator cache to the CURRENT new context
        trainer.observable_operator.cache = {}

    trainer.env.safe_set_context = safe_set_context

    # ---------------------------
    # 5. Generate trajectories
    # ---------------------------
    context_data = []
    st_data = []
    act_data = []
    obs_data = []
    reward_data = []

    for m in range(M):
        # Sample context first
        c = choice(trainer.env.contexts)
        trainer.env.safe_set_context(c)

        # Manual trajectory sampling (identical to sample_data but safe)
        st_list = []
        obs_list = []
        act_list = []
        traj_reward = 0

        # initial state for this context
        st = trainer.env.initial_states[0]
        st_list.append(st)
        traj_reward += trainer.env.reward_sampler(st)

        # initial observation from sensing action
        obs_list = []
        act = trainer.action_sampler_network(policy_net, obs_list)
        act_list.append(act)
        obs = trainer.env.observation_function_sampler(st, act)
        obs_list.append(obs)

        # unroll trajectory
        for t in range(T - 1):
            st = trainer.env.next_state_sampler(st, act)
            st_list.append(st)
            traj_reward += trainer.env.reward_sampler(st)

            act = trainer.action_sampler_network(policy_net, obs_list)
            act_list.append(act)

            obs = trainer.env.observation_function_sampler(st, act)
            obs_list.append(obs)

        # save trajectory
        context_data.append(c)
        st_data.append(st_list)
        act_data.append(act_list)
        obs_data.append(obs_list)
        reward_data.append(traj_reward)

    # ---------------------------
    # 6. Print trajectories
    # ---------------------------
    for m in range(M):
        a_list_idx = [trainer.env.actions.index(a) for a in act_data[m]]
        entropy = trainer.entopy_a_log_p_theta_obs(obs_data[m], a_list_idx)[0].item()
        print(f"Trajectory {m}")
        print("  context:     ", context_data[m])
        print("  states:      ", st_data[m])
        print("  actions:     ", act_data[m])
        print("  observations:", obs_data[m])
        print("  total reward:", reward_data[m])
        print("  H(C | y):    ", entropy)
        print("-" * 70)

    return {
        "context": context_data,
        "states": st_data,
        "obs": obs_data,
        "act": act_data,
        "reward": reward_data,
    }


if __name__ == "__main__":
    generate_trajectories_from_saved_policy(
        "../data_line_grid_1/Values/policy_net_2.pkl",
        M=50,
        T=10
    )
