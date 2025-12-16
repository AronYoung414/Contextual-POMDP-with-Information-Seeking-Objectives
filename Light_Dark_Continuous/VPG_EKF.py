import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math

EPS = 1e-300

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
    # def exact_P_y_given_c(self, context, z_seq, m_seq, act_seq):
    #     """
    #     Exact likelihood for Light–Dark model using Gaussian belief.
    #     """
    #
    #     # Initial Gaussian belief
    #     mu, var = self.env.initial_dist[context]
    #     Sigma = var
    #
    #     total_prob = 1.0
    #     log_prob = 0.0
    #
    #     obs_history = []
    #
    #     for t in range(self.T):
    #         a = act_seq[t]
    #         z = z_seq[t]
    #         m = m_seq[t]
    #
    #         # -----------------------------------------
    #         # POLICY TERM π(a|history)
    #         # -----------------------------------------
    #         if t == 0:
    #             inp = torch.zeros((1,1,2))
    #         else:
    #             inp = torch.tensor([obs_history], dtype=torch.float32)
    #
    #         with torch.no_grad():
    #             probs, _ = self.policy(inp)
    #         a_idx = self.env.actions.index(a)
    #         pi = float(probs[0, a_idx])
    #
    #         total_prob *= pi
    #         log_prob += math.log(pi)
    #
    #         # -----------------------------------------
    #         # STATE PREDICTION
    #         # -----------------------------------------
    #         if a == 'l':
    #             mu_pred = mu - self.env.step_size
    #             Sigma_pred = Sigma + self.env.process_noise
    #         elif a == 'r':
    #             mu_pred = mu + self.env.step_size
    #             Sigma_pred = Sigma + self.env.process_noise
    #         else:
    #             mu_pred = mu
    #             Sigma_pred = Sigma
    #         # Sigma_pred = Sigma
    #
    #         # ------------------------------------------
    #         # **ADD THIS: Transition likelihood term**
    #         # ------------------------------------------
    #         proc_var = self.env.process_noise
    #         trans_ll = (1.0 / math.sqrt(2 * math.pi * proc_var)) * \
    #                    math.exp(-(mu_pred - mu) ** 2 / (2 * proc_var))
    #         total_prob *= trans_ll
    #         log_prob += math.log(trans_ll)
    #
    #         # -----------------------------------------
    #         # OBSERVATION UPDATE (only when m=1)
    #         # -----------------------------------------
    #         if m == 1 and a == 'o':
    #             sigma_obs = self.env.obs_sigma(mu_pred, context)
    #             S = sigma_obs  # can be very close to 0.
    #             # print(z - mu_pred)
    #
    #             # observation likelihood
    #             obs_prob = (1.0 / math.sqrt(2*math.pi*S)) * \
    #                        math.exp(-(z - mu_pred)**2 / (2*S))
    #
    #             total_prob *= obs_prob
    #             log_prob += math.log(obs_prob+EPS)
    #
    #             # Kalman update
    #             K = Sigma_pred / S
    #             mu = mu_pred + K * (z - mu_pred)
    #             Sigma = (1 - K) * Sigma_pred
    #
    #         else:
    #             # No observation: belief stays predicted
    #             mu = mu_pred
    #             Sigma = Sigma_pred
    #
    #         # -----------------------------------------
    #         # update observation history (for policy)
    #         # -----------------------------------------
    #         obs_history.append([z, m])
    #         # print(total_prob, log_prob)
    #
    #     return total_prob, log_prob

    def exact_P_y_given_c(self, context, z_seq, m_seq, act_seq):

        mu, var = self.env.initial_dist[context]
        Sigma = var

        total_prob = 1.0
        log_prob = 0.0

        obs_history = []

        for t in range(self.T):

            a = act_seq[t]
            z = z_seq[t]
            m = m_seq[t]

            # ---------------------------
            # POLICY likelihood
            # ---------------------------
            if t == 0:
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            with torch.no_grad():
                probs, _ = self.policy(inp)

            a_idx = self.env.actions.index(a)
            pi = float(probs[0, a_idx])

            log_prob += math.log(pi)
            total_prob *= pi

            # ---------------------------
            # STATE prediction
            # ---------------------------
            if a == 'l':
                mu_pred = mu - self.env.step_size
                Sigma_pred = Sigma + self.env.process_noise
            elif a == 'r':
                mu_pred = mu + self.env.step_size
                Sigma_pred = Sigma + self.env.process_noise
            else:
                mu_pred = mu
                Sigma_pred = Sigma

            # ---------------------------
            # OBSERVATION likelihood
            # ---------------------------
            if m == 1 and a == 'o':
                sigma_obs = self.env.obs_sigma(mu_pred, context)

                obs_prob = (1.0 / math.sqrt(2 * math.pi * sigma_obs)) * \
                           math.exp(-(z - mu_pred) ** 2 / (2 * sigma_obs))

                log_prob += math.log(obs_prob + EPS)
                total_prob *= obs_prob

                # EKF update
                K = Sigma_pred / (Sigma_pred + sigma_obs)
                mu = mu_pred + K * (z - mu_pred)
                Sigma = (1 - K) * Sigma_pred

            else:
                mu = mu_pred
                Sigma = Sigma_pred

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
        # print(sum(Pw))
        P_y = sum(Pw)
        logP = math.log(P_y+EPS)
        # print(P_y, logP)
        return logP

    # ------------------------------------------------------------------
    # 3. Entropy H(C|y)
    # ------------------------------------------------------------------
    def entropy_C_given_y(self, z_seq, m_seq, act_seq, m=11):
        numerators = []
        for c in self.env.contexts:
            Pygc, _ = self.exact_P_y_given_c(c, z_seq, m_seq, act_seq)
            numerators.append(self.env.context_distribution[c] * Pygc)

        Z = sum(numerators) + EPS
        post = [n/Z for n in numerators]
        # if m < 10:
        #     print(post)

        H = 0.0
        for pcy in post:
            H += -pcy * math.log2(pcy+EPS)  # discrete entropy
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
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            probs, _ = self.policy(inp)
            a_idx = self.env.actions.index(a)
            log_terms.append(torch.log2(probs[0, a_idx]))

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

        x_seq = []
        z_seq = []
        m_seq = []
        act_seq = []
        reward = 0.0

        obs_history = []
        x_seq.append(x)

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
            x_seq.append(x)
            reward += r

            if z is None:
                z_seq.append(0.0)
                m_seq.append(0.0)
            else:
                z_seq.append(float(z))
                m_seq.append(1.0)

            obs_history.append([z_seq[-1], m_seq[-1]])

        return context, x_seq, z_seq, m_seq, act_seq, reward

    # ------------------------------------------------------------------
    # 6. Train step
    # ------------------------------------------------------------------
    def train_step(self, M):

        total_R = 0
        total_H = 0
        total_V = 0
        total_logPy = 0  # for debugging
        total_RTau = 0  # for debugging

        accumulated = None

        for m in range(M):

            context, x_seq, z_seq, m_seq, act_seq, R = self.sample_trajectory()
            # if m < 10:
            #     print(context)
            #     print(x_seq)
            #     print(z_seq)
            #     print(m_seq)
            #     print(act_seq)

            # exact likelihood
            logP_y = self.P_and_logP_y(z_seq, m_seq, act_seq)

            # posterior entropy
            H = self.entropy_C_given_y(z_seq, m_seq, act_seq, m)

            # gradient
            grad_log = self.grad_log_P_y_given_c(z_seq, m_seq, act_seq)

            # print(logP_y, R)
            # weight
            cst = logP_y - (R / self.tau)

            total_R += R
            total_H += H
            total_V += (R - self.tau * H)
            total_logPy += logP_y
            total_RTau += R / self.tau

            # accumulate gradients
            if accumulated is None:
                accumulated = [cst * g for g in grad_log]
            else:
                for i in range(len(grad_log)):
                    accumulated[i] += cst * grad_log[i]

        # print(total_logPy, total_RTau)
        # apply gradients
        for i, p in enumerate(self.policy.parameters()):
            p.grad = accumulated[i] / M

        self.optim.step()
        self.optim.zero_grad()

        # return metrics
        return total_H / M, total_R / M, total_V / M
