# run_pf_test.py

import torch
from light_dark_environment import ContinuousLightDarkPOMDP
from policy_continuous import ContinuousPolicyNetworkMasked
from particle_filter import ParticleFilter
from exact_likelihood import exact_likelihood


def generate_trajectory(env, context, act_seq):
    x = env.sample_initial_state(context)

    z_seq = []
    m_seq = []  # mask seq

    for a in act_seq:
        x, z, r = env.step(x, a, context)

        if z is None:
            z_seq.append(0.0)  # dummy numeric
            m_seq.append(0.0)  # mask = 0 means "no observation"
        else:
            z_seq.append(float(z))
            m_seq.append(1.0)  # real observation

    return z_seq, m_seq


# def main():
#     env = ContinuousLightDarkPOMDP()
#     policy = ContinuousPolicyNetworkMasked(action_size=3)
#
#     pf = ParticleFilter(env, policy, num_particles=3000)
#
#     context = 0
#     act_seq = ['l', 'l', 'o', 'r', 'o', 'l', 'o']  # mixed movement & sensing
#
#     z_seq, m_seq = generate_trajectory(env, context, act_seq)
#
#     print("z_seq:", z_seq)
#     print("m_seq:", m_seq)
#
#     log_pf = pf.compute_trajectory_prob(context, z_seq, m_seq, act_seq)
#     log_exact = exact_likelihood(env, context, z_seq, m_seq, act_seq, policy)
#
#     print("PF:", log_pf)
#     print("Exact:", log_exact)
#     print("Error:", abs(log_pf - log_exact))
#
#
# if __name__ == "__main__":
#     main()
