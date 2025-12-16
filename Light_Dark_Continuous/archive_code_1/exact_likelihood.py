# exact_likelihood.py

import math
import torch


def exact_likelihood(env, context, z_seq, m_seq, act_seq, policy):
    mu, var = env.initial_dist[context]
    Sigma = var

    log_prob = 0.0
    obs_history = []

    for t, a in enumerate(act_seq):
        z = z_seq[t]
        m = m_seq[t]

        # policy term
        if len(obs_history) == 0:
            obs_tensor = torch.zeros((1, 1, 2))
        else:
            obs_tensor = torch.tensor([obs_history], dtype=torch.float32)

        probs, _ = policy(obs_tensor)
        a_idx = env.actions.index(a)
        log_prob += math.log(float(probs[0][a_idx]) + 1e-12)

        # predict
        if a == 'l':
            mu_pred = mu - env.step_size
        elif a == 'r':
            mu_pred = mu + env.step_size
        else:
            mu_pred = mu

        # Sigma_pred = Sigma + env.process_noise
        Sigma_pred = Sigma

        # observation update only if m=1
        if m == 1:
            sigma_obs = env.obs_sigma(mu_pred)
            S = Sigma_pred + sigma_obs ** 2

            ll = (1.0 / math.sqrt(2 * math.pi * S)) * math.exp(-(z - mu_pred) ** 2 / (2 * S))
            log_prob += math.log(ll + 1e-12)

            K = Sigma_pred / S
            mu = mu_pred + K * (z - mu_pred)
            Sigma = (1 - K) * Sigma_pred
        else:
            # no update
            mu = mu_pred
            Sigma = Sigma_pred

        obs_history.append([z, m])

    return log_prob
