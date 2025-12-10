import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math


class VariationalPolicyGradientContinuousPF:
    """
    Continuous Light–Dark VPG using probability-first Particle Filter.

    Gradient:
        ∇θ L = E[(logPθ(y) − R/τ) ∇θ logPθ(y|c)]
    """

    def __init__(self, env, policy_net, pf, tau, horizon, lr=1e-3):
        self.env = env
        self.policy = policy_net
        self.pf = pf
        self.tau = tau
        self.T = horizon
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

    # -------------------------------------------------------
    # 1. probability & logP(y | c)
    # -------------------------------------------------------
    def P_y_given_c(self, context, z_seq, m_seq, act_seq):
        P, logP = self.pf.compute_trajectory_prob(context, z_seq, m_seq, act_seq)
        return P, logP

    # -------------------------------------------------------
    # 2. log P(y) = log Σ_c [ p(c) P(y|c) ]
    # -------------------------------------------------------
    def log_p_y(self, z_seq, m_seq, act_seq):
        weighted_probs = []
        for c in self.env.contexts:
            P, _ = self.P_y_given_c(c, z_seq, m_seq, act_seq)
            weighted_probs.append(self.env.context_distribution[c] * P)

        totalP = sum(weighted_probs) + 1e-12
        return math.log(totalP)

    # -------------------------------------------------------
    # 3. H(C | y) using probability-first PF
    # -------------------------------------------------------
    def entropy_C_given_y(self, z_seq, m_seq, act_seq):
        """
        H(C|y) = - Σ_c p(c|y) log p(c|y)
        where p(c|y) = p(c)*P(y|c) / Σ_c p(c)P(y|c)
        """
        numerators = []
        for c in self.env.contexts:
            P, _ = self.P_y_given_c(c, z_seq, m_seq, act_seq)
            numerators.append(self.env.context_distribution[c] * P)

        Z = sum(numerators) + 1e-12
        post = [n / Z for n in numerators]

        H = 0.0
        for pcy in post:
            H += -pcy * math.log(pcy + 1e-12)

        return H

    # -------------------------------------------------------
    # 4. ∇θ log Pθ(y|c)
    # -------------------------------------------------------
    def grad_log_p_y_given_c(self, z_seq, m_seq, act_seq):
        """
        Gradient of log policy only:
            Σ log π(a_t | history)
        PF never appears in gradient.
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

    # -------------------------------------------------------
    # 5. SAMPLE TRAJECTORY (continuous)
    # -------------------------------------------------------
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
                inp = torch.zeros((1,1,2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = self.policy(inp)
                a = np.random.choice(self.env.actions, p=probs[0].numpy())

            act_seq.append(a)

            # environment step
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

    # -------------------------------------------------------
    # 6. ONE TRAINING STEP
    # -------------------------------------------------------
    def train_step(self, M):

        total_R = 0.0
        total_H = 0.0
        total_value = 0.0

        accumulated = None

        for _ in range(M):

            context, z_seq, m_seq, act_seq, R = self.sample_trajectory()

            # likelihood of y under θ (marginal over contexts)
            logP = self.log_p_y(z_seq, m_seq, act_seq)

            # posterior entropy
            H = self.entropy_C_given_y(z_seq, m_seq, act_seq)

            # gradient of logP(y|c)
            grad_log = self.grad_log_p_y_given_c(z_seq, m_seq, act_seq)

            # coefficient
            cst = logP - (R / self.tau)

            total_R += R
            total_H += H
            total_value += (R - H)

            # accumulate gradients
            if accumulated is None:
                accumulated = [cst * g for g in grad_log]
            else:
                for i in range(len(grad_log)):
                    accumulated[i] += cst * grad_log[i]

        # apply gradient
        for i, p in enumerate(self.policy.parameters()):
            p.grad = accumulated[i] / M

        self.optim.step()
        self.optim.zero_grad()

        # return averages
        return total_H / M, total_R / M, total_value / M
