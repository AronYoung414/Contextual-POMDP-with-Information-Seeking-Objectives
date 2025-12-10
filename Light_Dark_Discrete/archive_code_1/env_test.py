from random import choices
from random import choice
from line_grid_environment import Environment


def sample_random_data(env, M, T):
    st_data = []
    act_data = []
    obs_data = []
    for m in range(M):
        st_list = []
        act_list = []
        obs_list = []
        st = env.initial_states[0]
        # Sample sensing action
        act = choice(env.actions)
        act_list.append(act)
        # Get the observation of initial state
        obs = env.observation_function_sampler(st, act)
        obs_list.append(obs)

        for t in range(T - 1):
            st_list.append(st)
            # sample the next state
            st = env.next_state_sampler(st, act)
            # Sample sensing action
            act = choice(env.actions)
            act_list.append(act)
            # Add the observation
            obs = env.observation_function_sampler(st, act)
            obs_list.append(obs)

        st_data.append(st_list)
        obs_data.append(obs_list)
        act_data.append(act_list)
    return st_data, obs_data, act_data


def display_data(s_data, y_data, a_data):
    M = len(s_data)
    for k in range(M):
        print(s_data[k])
        print(y_data[k])
        print(a_data[k])
    return 0


M = 20  # number of sampled trajectories
T = 10  # length of a trajectory
line_grid = Environment(0, 0.1, 0.1)
s_data, y_data, a_data = sample_random_data(line_grid, M, T)
display_data(s_data, y_data, a_data)
