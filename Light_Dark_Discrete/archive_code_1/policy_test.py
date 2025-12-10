from variational_policy_gradient import *


def generate_trajectories_from_saved_policy(policy_path):
    # 1) Set seeds for reproducibility
    set_random_seeds(1337)

    # 2) Rebuild environment (same as training setup)
    exp_name = 'line_grid_1'
    env = Environment(choice([0, 1]), 0.1, 0.1)

    # 3) Rebuild the policy network with the same dimensions as during training
    policy_net = PolicyNetwork(
        obs_vocab_size=len(env.observations),
        action_size=env.action_size,   # env.action_size == 3 here
        hidden_dim=64,
        max_seq_len=10
    )

    # 4) Load the pretrained weights
    # Change this path to wherever your trained policy is stored
    state_dict = torch.load(policy_path, map_location="cpu")
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    # 5) Wrap the policy into your VariationalPolicyGradient class
    #    We won't call .train(), just use the sampling code.
    trainer = VariationalPolicyGradient(
        exp_name=exp_name,
        env=env,
        experiment_num=2,
        tau=0.1,
        gamma=1,
        horizon=10,          # T: trajectory length
        trajectories_num=20,  # M: not used here if we pass M manually
        iteration_num=0,     # no training iterations
        step_size=0.001,
        policy_network=policy_net
    )

    # 6) Generate trajectories using the loaded policy
    M = 50      # number of trajectories you want
    T = 10      # horizon length (must match env/hyperparams)
    context_data, st_data, obs_data, act_data, reward_data = trainer.sample_data(M, T)

    # 7) Example: print trajectories or save them to a file
    for m in range(M):
        a_list = [trainer.env.actions.index(act) for act in act_data[m]]
        entropy = trainer.entopy_a_log_p_theta_obs(obs_data[m], a_list)[0].item()
        print(f"Trajectory {m}")
        print("  context:     ", context_data[m])
        print("  states:      ", st_data[m])
        print("  actions:     ", act_data[m])
        print("  observations:", obs_data[m])
        print("  total reward:", reward_data[m])
        print("  Trajectory entropy:", entropy)
        print("-" * 60)

    # If you prefer to save them:
    # import pickle
    # with open("generated_trajectories.pkl", "wb") as f:
    #     pickle.dump(
    #         {
    #             "context": context_data,
    #             "states": st_data,
    #             "obs": obs_data,
    #             "act": act_data,
    #             "reward": reward_data,
    #         },
    #         f,
    #     )


if __name__ == "__main__":
    # Option A: train from scratch
    # main()

    # Option B: comment out training and just generate trajectories
    generate_trajectories_from_saved_policy("data_line_grid_1/Values/policy_net_2.pkl")

