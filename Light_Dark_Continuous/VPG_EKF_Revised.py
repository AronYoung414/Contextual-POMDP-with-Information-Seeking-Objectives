import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import math


class VariationalPolicyGradientEKF:
    """
    Variational Policy Gradient for Contextual POMDP using
    EKF-compatible likelihood approximation.

    Implements the gradient:
        ∇θ L = E[(log Pθ(y) - R/τ) ∇θ log Pθ(y|c)]

    Where:
        log Pθ(y|c) = Σ_t log πθ(a_t|history)
                      + Σ_t log p_obs(z_t | μ_pred, c)
                      + (optional) transition likelihood terms
    """

    def __init__(self, env, policy_net, tau, horizon, lr=1e-3):
        self.env = env
        self.policy = policy_net
        self.tau = tau
        self.T = horizon

        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

    # ============================================================
    # 1. EKF-COMPATIBLE log P(y | c)
    # ============================================================
    def log_P_y_given_c(self, context, z_seq, m_seq, act_seq):
        """
        Compute EKF-compatible approximate likelihood.

        Important:
        - Uses generative observation noise σ_obs(context, μ_pred)
        - Does NOT include EKF covariance in the likelihood
        - EKF covariance is only for filtering, not likelihood
        """

        # initial belief
        mu, var = self.env.initial_dist[context]
        Sigma = var  # used for FILTERING only

        logp = 0.0
        obs_history = []

        for t in range(self.T):

            a = act_seq[t]

            # ------------------------------------------
            # 1) Policy likelihood term: log π(a_t | ...)
            # ------------------------------------------
            if t == 0:
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = self.policy(inp)

            a_idx = self.env.actions.index(a)
            pi_t = float(probs[0, a_idx])

            logp += math.log(pi_t + 1e-12)

            # ------------------------------------------
            # 2) EKF PREDICTION step
            # ------------------------------------------
            if a == 'l':
                mu_pred = mu - self.env.step_size
            elif a == 'r':
                mu_pred = mu + self.env.step_size
            else:
                mu_pred = mu

            Sigma_pred = Sigma + self.env.process_noise

            # ------------------------------------------
            # 3) OBSERVATION likelihood (generative model)
            # ------------------------------------------
            z = z_seq[t]
            m = m_seq[t]

            if m == 1 and a == 'o':
                # generative-model noise based on predicted mean
                sigma_obs = self.env.obs_sigma(mu_pred, context)
                var_obs = sigma_obs**2

                # GENERATIVE observation likelihood (correct!)
                obs_ll = (1.0 / math.sqrt(2 * math.pi * var_obs)) * \
                         math.exp(-(z - mu_pred) ** 2 / (2 * var_obs))

                logp += math.log(obs_ll + 1e-12)

                # EKF UPDATE step (affects μ, Σ ONLY)
                K = Sigma_pred / (Sigma_pred + var_obs)
                mu = mu_pred + K * (z - mu_pred)
                Sigma = (1 - K) * Sigma_pred

            else:
                # no observation
                mu = mu_pred
                Sigma = Sigma_pred

            # ------------------------------------------
            # 4) Store for policy RNN input
            # ------------------------------------------
            obs_history.append([z, m])

        return logp

    # ============================================================
    # 2. Compute log P(y) = log Σ_c p(c) P(y|c)
    # ============================================================
    def log_P_y(self, z_seq, m_seq, act_seq):
        vals = []
        for c in self.env.contexts:
            logp_c = self.log_P_y_given_c(c, z_seq, m_seq, act_seq)
            vals.append(math.log(self.env.context_distribution[c] + 1e-12) +
                        logp_c)

        # log-sum-exp for numerical stability
        mmax = max(vals)
        return mmax + math.log(sum(math.exp(v - mmax) for v in vals) + 1e-12)

    # ============================================================
    # 3. Gradient of log P(y | c)
    # ============================================================
    def grad_log_P_y_given_c(self, z_seq, m_seq, act_seq):
        """
        ∇θ log Pθ(y|c) = Σ_t ∇θ log πθ(a_t | history)
        """

        self.policy.zero_grad()
        log_terms = []
        obs_history = []

        for t, a in enumerate(act_seq):

            if t == 0:
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            probs, _ = self.policy(inp)
            a_idx = self.env.actions.index(a)

            log_terms.append(torch.log(probs[0, a_idx] + 1e-12))

            obs_history.append([z_seq[t], m_seq[t]])

        total_log = torch.stack(log_terms).sum()
        total_log.backward()

        grads = []
        for p in self.policy.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p))
            else:
                grads.append(p.grad.clone())

        return grads

    # ============================================================
    # POSTERIOR ENTROPY H(C | y)
    # ============================================================
    def posterior_entropy(self, z_seq, m_seq, act_seq):
        """
        Computes posterior entropy:
            H(C|y) = - sum_c p(c|y) log p(c|y)
        using the EKF likelihood P(y|c).
        """
        logps = []
        for c in self.env.contexts:
            logp_c = self.log_P_y_given_c(c, z_seq, m_seq, act_seq)
            logps.append(math.log(self.env.context_distribution[c] + 1e-12) + logp_c)

        # log-sum-exp for denominator
        mmax = max(logps)
        denom = sum(math.exp(v - mmax) for v in logps) + 1e-12

        # compute posterior probabilities
        post = [(math.exp(v - mmax) / denom) for v in logps]

        # entropy
        H = -sum(p * math.log(p + 1e-12) for p in post)
        return H

    # ============================================================
    # 4. Sample one trajectory
    # ============================================================
    def sample_trajectory(self):
        context = np.random.choice(self.env.contexts)
        x = self.env.sample_initial_state(context)

        z_seq = []
        m_seq = []
        act_seq = []
        reward = 0.0

        obs_history = []

        for t in range(self.T):
            # RNN policy input
            if t == 0:
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = self.policy(inp)

            a = np.random.choice(self.env.actions, p=probs[0].numpy())
            act_seq.append(a)

            # Environment step
            x, z, r = self.env.step(x, a, context)
            reward += r

            if z is None:
                z_seq.append(0.0)
                m_seq.append(0.0)
            else:
                z_seq.append(float(z))
                m_seq.append(1.0)

            obs_history.append([z_seq[-1], m_seq[-1]])

        return context, z_seq, m_seq, act_seq, reward

    # ============================================================
    # 5. ONE TRAINING STEP (M trajectories)
    # ============================================================
    # ============================================================
    # TRAINING STEP: returns (H_avg, R_avg, V_avg)
    # ============================================================
    # ============================================================
    # TRAINING STEP: returns (H_avg, R_avg, V_avg)
    # ============================================================
    # ============================================================
    # TRAINING STEP: returns (H_avg, R_avg, V_avg)
    # ============================================================
    def train_step(self, M):

        total_R = 0
        total_V = 0
        total_H = 0

        accumulated = None

        for _ in range(M):

            context, z_seq, m_seq, act_seq, R = self.sample_trajectory()

            # ----- log P(y) -----
            logP_y = self.log_P_y(z_seq, m_seq, act_seq)

            # ----- posterior entropy H(C|y) -----
            H = self.posterior_entropy(z_seq, m_seq, act_seq)

            # ----- ∇ log P(y | c) -----
            grad_log = self.grad_log_P_y_given_c(z_seq, m_seq, act_seq)

            # ----- VPG weighting term -----
            weight = logP_y - (R / self.tau)

            total_R += R
            total_V += (R - self.tau * H)
            total_H += H

            # ----- Accumulate gradients -----
            if accumulated is None:
                accumulated = [weight * g for g in grad_log]
            else:
                for i in range(len(grad_log)):
                    accumulated[i] += weight * grad_log[i]

        # ----- Apply gradient -----
        for i, p in enumerate(self.policy.parameters()):
            p.grad = accumulated[i] / M

        self.optim.step()
        self.optim.zero_grad()

        # ----- Return ALL THREE -----
        return total_H / M, total_R / M, total_V / M


