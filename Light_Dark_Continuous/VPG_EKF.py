import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math


class VariationalPolicyGradientExact:
    """
    VPG using EXACT likelihood (Kalman-style) for the Light–Dark CPOMDP.

    Computes:
        P(y | c), log P(y | c)
        P(y), log P(y)
        H(C | y)

    And gradient:
        ∇θ log Pθ(y | c) = Σ_t ∇θ log πθ(a_t | o_1:t)
    """

    def __init__(self, env, policy_net, tau, horizon, lr=1e-3):
        self.env = env
        self.policy = policy_net
        self.tau = tau
        self.T = horizon
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 1. Compute exact likelihood P(y|c) and log P(y|c)
    # ------------------------------------------------------------------
    def exact_P_y_given_c(self, context, z_seq, m_seq, act_seq):
        """
        Exact likelihood for Light–Dark model using Gaussian belief.
        """

        # Initial Gaussian belief
        mu, var = self.env.initial_dist[context]
        Sigma = var

        total_prob = 1.0
        log_prob = 0.0

        obs_history = []

        for t in range(self.T):
            a = act_seq[t]
            z = z_seq[t]
            m = m_seq[t]

            # -----------------------------------------
            # POLICY TERM π(a|history)
            # -----------------------------------------
            if t == 0:
                inp = torch.zeros((1,1,2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = self.policy(inp)
            a_idx = self.env.actions.index(a)
            pi = float(probs[0, a_idx])

            total_prob *= pi
            log_prob += math.log(pi + 1e-12)

            # -----------------------------------------
            # STATE PREDICTION
            # -----------------------------------------
            if a == 'l':
                mu_pred = mu - self.env.step_size
            elif a == 'r':
                mu_pred = mu + self.env.step_size
            else:
                mu_pred = mu

            Sigma_pred = Sigma + self.env.process_noise
            # Sigma_pred = Sigma**2

            # -----------------------------------------
            # OBSERVATION UPDATE (only when m=1)
            # -----------------------------------------
            if m == 1 and a == 'o':
                sigma_obs = self.env._obs_sigma(mu_pred, context)
                S = Sigma_pred + sigma_obs

                # observation likelihood
                obs_prob = (1.0 / math.sqrt(2*math.pi*S)) * \
                           math.exp(-(z - mu_pred)**2 / (2*S))

                total_prob *= obs_prob
                log_prob += math.log(obs_prob + 1e-12)

                # Kalman update
                K = Sigma_pred / S
                mu = mu_pred + K * (z - mu_pred)
                Sigma = (1 - K) * Sigma_pred

            else:
                # No observation: belief stays predicted
                mu = mu_pred
                Sigma = Sigma_pred

            # -----------------------------------------
            # update observation history (for policy)
            # -----------------------------------------
            obs_history.append([z, m])

        return total_prob, log_prob

    # ------------------------------------------------------------------
    # 2. Compute P(y) = Σ p(c)P(y|c)  and log P(y)
    # ------------------------------------------------------------------
    def P_and_logP_y(self, z_seq, m_seq, act_seq):
        Pw = []
        for c in self.env.contexts:
            Pc, _ = self.exact_P_y_given_c(c, z_seq, m_seq, act_seq)
            Pw.append(self.env.context_distribution[c] * Pc)

        P_y = sum(Pw) + 1e-12
        logP = math.log(P_y)
        return P_y, logP

    # ------------------------------------------------------------------
    # 3. Entropy H(C|y)
    # ------------------------------------------------------------------
    def entropy_C_given_y(self, z_seq, m_seq, act_seq):
        numerators = []
        for c in self.env.contexts:
            Pygc, _ = self.exact_P_y_given_c(c, z_seq, m_seq, act_seq)
            numerators.append(self.env.context_distribution[c] * Pygc)

        Z = sum(numerators) + 1e-12
        post = [n/Z for n in numerators]

        H = 0.0
        for pcy in post:
            H += -pcy * math.log(pcy + 1e-12)
        return H

    # ------------------------------------------------------------------
    # 4. Gradient ∇θ log P(y|c) = Σ ∇θ log π(a_t)
    # ------------------------------------------------------------------
    def grad_log_P_y_given_c(self, z_seq, m_seq, act_seq):
        self.policy.zero_grad()
        log_terms = []
        obs_history = []

        for t, a in enumerate(act_seq):

            if t == 0:
                inp = torch.zeros((1,1,2))
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

    # ------------------------------------------------------------------
    # 5. Sampling trajectories
    # ------------------------------------------------------------------
    def sample_trajectory(self):
        context = np.random.choice(self.env.contexts)
        x = self.env.sample_initial_state(context)

        z_seq = []
        m_seq = []
        act_seq = []
        reward = 0.0

        obs_history = []

        for t in range(self.T):
            # policy input
            if t == 0:
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = self.policy(inp)
            a = np.random.choice(self.env.actions, p=probs[0].numpy())
            act_seq.append(a)

            # transition
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

    # ------------------------------------------------------------------
    # 6. Train step
    # ------------------------------------------------------------------
    def train_step(self, M):

        total_R = 0
        total_H = 0
        total_V = 0

        accumulated = None

        for _ in range(M):

            context, z_seq, m_seq, act_seq, R = self.sample_trajectory()

            # exact likelihood
            P_y, logP_y = self.P_and_logP_y(z_seq, m_seq, act_seq)

            # posterior entropy
            H = self.entropy_C_given_y(z_seq, m_seq, act_seq)

            # gradient
            grad_log = self.grad_log_P_y_given_c(z_seq, m_seq, act_seq)

            # weight
            cst = logP_y - (R / self.tau)

            total_R += R
            total_H += H
            total_V += (R - self.tau * H)

            # accumulate gradients
            if accumulated is None:
                accumulated = [cst * g for g in grad_log]
            else:
                for i in range(len(grad_log)):
                    accumulated[i] += cst * grad_log[i]

        # apply gradients
        for i, p in enumerate(self.policy.parameters()):
            p.grad = accumulated[i] / M

        self.optim.step()
        self.optim.zero_grad()

        # return metrics
        return total_H / M, total_R / M, total_V / M
