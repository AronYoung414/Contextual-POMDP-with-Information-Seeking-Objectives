import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from intersection_env import VisionCtxIntersectionEnv
from GRU_policy import obs_to_tensor

EPS = 1e-300


# ============================================================
# Recursive Bayesian Context Filter
# ============================================================

class RecursiveContextFilter:
    """
    Implements the theory-faithful recursion:
      b_t(c) ∝ b_{t-1}(c) * P(o_t | h_{t-1}, c)
      logZ_T = Σ_t log Σ_c b_{t-1}(c) P(o_t | h_{t-1}, c)
      posterior = b_T
      entropy = H(C|y)

    You provide a predictive log-likelihood model:
      log_obs_likelihood(obs_t, obs_tm1, act_tm1, c) ≈ log P(o_t | h_{t-1}, c)
    """

    def __init__(self, contexts, prior_dict):
        self.contexts = list(contexts)
        self.prior = dict(prior_dict)

    def log_obs_likelihood(self, obs_t, obs_tm1, c, env, t):
        # 1. If NPC not visible → no information
        if np.any(np.isnan(obs_t[4:])):
            return 0.0  # log(1)

        # 2. If no previous NPC observation → weak evidence
        if obs_tm1 is None or np.any(np.isnan(obs_tm1[4:])):
            return 0.0

        # 3. Use PAST agent & NPC state
        ax, ay = obs_tm1[0], obs_tm1[1]
        nx, ny, nvx, nvy = obs_tm1[4:]

        if c == 1:
            acc = env._npc_normal_potential_acc(nx, ny, nvx, nvy, ax, ay)
            sigma0 = 0.6
        else:
            acc = env._npc_aggressive_potential_acc(nx, ny, nvx, nvy, ax, ay)
            sigma0 = 0.2

        dt = env.dt
        nvx_p = nvx + dt * acc[0]
        nvy_p = nvy + dt * acc[1]
        nx_p = nx + dt * nvx_p
        ny_p = ny + dt * nvy_p

        mu = np.array([nx_p, ny_p, nvx_p, nvy_p])
        z = np.array(obs_t[4:])

        # 4. Time-growing covariance (EKF analogue)
        sigma = np.sqrt(sigma0 ** 2 + t * env.sigma_dyn ** 2)

        diff = z - mu
        ll = -0.5 * np.sum(diff ** 2) / (sigma ** 2)
        ll += -4.0 * np.log(sigma + 1e-12)

        return ll

    def run_filter(self, obs_seq, env):
        """
        Runs recursion and returns:
          logZ   : log P(o_{0:T} | a_{0:T-1})
          log_post: dict c -> log P(c | y)
          entropy: H(C|y) in bits
          post_probs: dict c -> P(c|y)
        """
        # initialize b_{-1}(c) = prior
        log_b = {c: math.log(self.prior[c] + EPS) for c in self.contexts}
        logZ = 0.0

        for t in range(len(obs_seq)):
            obs_t = obs_seq[t]
            obs_tm1 = obs_seq[t - 1] if t > 0 else None

            # log predictive likelihood for each context
            log_pred = {}
            for c in self.contexts:
                log_pred[c] = self.log_obs_likelihood(obs_t, obs_tm1, c, env, t)

            # ell_t = log Σ_c exp(log_b(c) + log_pred(c))
            m = max(log_b[c] + log_pred[c] for c in self.contexts)
            sum_exp = sum(math.exp(log_b[c] + log_pred[c] - m) for c in self.contexts)
            ell_t = m + math.log(sum_exp + EPS)

            # logZ_t = logZ_{t-1} + ell_t
            logZ += ell_t

            # belief update: log b_t(c) = log b_{t-1}(c) + log_pred(c) - ell_t
            for c in self.contexts:
                log_b[c] = log_b[c] + log_pred[c] - ell_t

        # posterior at final time
        log_post = dict(log_b)

        post_probs = {c: math.exp(log_post[c]) for c in self.contexts}
        # numerical normalize
        Zp = sum(post_probs.values()) + EPS
        for c in self.contexts:
            post_probs[c] /= Zp
            log_post[c] = math.log(post_probs[c] + EPS)

        entropy = 0.0
        for c in self.contexts:
            p = post_probs[c]
            entropy += -p * math.log2(p + EPS)

        return logZ, log_post, entropy, post_probs


# ============================================================
# VPG Trainer (keeps your algorithmic structure)
# ============================================================

class VPGContextFilterTrainer:
    def __init__(self, env, policy, tau=0.2, lr=1e-3, T=250, device=None, seed=0):
        self.env_template = env          # used only for params (img_size, camera_range, etc.)
        self.policy = policy
        self.tau = float(tau)
        self.T = int(T)
        self.seed = int(seed)

        self.device = device or next(policy.parameters()).device
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

        self.contexts = [1, 2]
        self.context_prior = {1: 0.5, 2: 0.5}
        self.filter = RecursiveContextFilter(self.contexts, self.context_prior)

    def _make_env_for_ctx(self, ctx: int):
        # Important: your env fixes ctx at construction
        return VisionCtxIntersectionEnv(
            ctx=ctx,
            img_size=getattr(self.env_template, "img_size", 96),
            horizon=self.T,
            camera_range=getattr(self.env_template, "camera_range", 10.0),
        )

    # --------------------------------------------------------
    # 5. Sampling trajectories (same spirit as your code)
    # --------------------------------------------------------
    def sample_trajectory(self):
        ctx = np.random.choice(self.contexts)
        env = self._make_env_for_ctx(ctx)

        obs, _ = env.reset(seed=self.seed)
        self.policy.reset_hidden()

        obs_seq = []
        act_seq = []
        logpi_seq = []
        reward = 0.0

        for t in range(self.T):
            obs_t = obs_to_tensor(obs, self.device)  # (16,)

            a, logp = self.policy.sample_and_log_prob(obs_t)
            action_np = a.detach().cpu().numpy().astype(np.float32)

            next_obs, r, done, trunc, _ = env.step(action_np)

            obs_seq.append(obs.copy())
            act_seq.append(action_np.copy())
            logpi_seq.append(float(logp.detach().cpu().item()))

            reward += float(r)
            obs = next_obs

            if done or trunc:
                break

        return ctx, env, obs_seq, act_seq, logpi_seq, reward

    # --------------------------------------------------------
    # 4. Gradient ∇θ log P(y|c) = Σ ∇θ log π(a_t | h_t)
    # --------------------------------------------------------
    def grad_log_P_y_given_c(self, obs_seq):
        """
        Returns grads of Σ_t log π(a_t | h_t).
        We recompute actions/logprobs with rsample() in the policy; this matches
        the structure of your old code (stochastic grad estimate each pass).
        """
        self.policy.zero_grad()
        self.policy.reset_hidden()

        log_terms = []
        for t in range(len(obs_seq)):
            obs_t = obs_to_tensor(obs_seq[t], self.device)
            _, logp = self.policy.sample_and_log_prob(obs_t)
            log_terms.append(logp)

        total_log = torch.stack(log_terms).sum()
        total_log.backward()

        grads = []
        for p in self.policy.parameters():
            grads.append(torch.zeros_like(p) if p.grad is None else p.grad.clone())
        return grads

    # --------------------------------------------------------
    # 6. Train step (same functionality / structure)
    # --------------------------------------------------------
    def train_step(self, M: int):
        total_R = 0.0
        total_H = 0.0
        total_V = 0.0

        total_logPobs = 0.0
        total_post_c1 = 0.0

        accumulated = None

        for m in range(M):
            ctx, env, obs_seq, act_seq, logpi_seq, R = self.sample_trajectory()

            # --- Recursive filter: logZ = log P(o | a), posterior, entropy
            logP_obs_given_a, log_post, H, post_probs = self.filter.run_filter(
                obs_seq=obs_seq,
                env=env,
            )

            # If you want log P(y) including actions, add Σ logπ:
            sum_logpi = float(np.sum(logpi_seq))
            logP_y = logP_obs_given_a + sum_logpi

            # --- Gradient term (policy only)
            grad_log = self.grad_log_P_y_given_c(obs_seq)

            # --- Weight (same as your original code)
            cst = logP_y - (R / self.tau)

            total_R += R
            total_H += H
            total_V += (R - self.tau * H)
            total_logPobs += logP_y
            total_post_c1 += post_probs[1]

            # accumulate gradients
            if accumulated is None:
                accumulated = [cst * g for g in grad_log]
            else:
                for i in range(len(grad_log)):
                    accumulated[i] += cst * grad_log[i]

        # apply gradients
        for i, p in enumerate(self.policy.parameters()):
            p.grad = accumulated[i] / float(M)

        self.optim.step()
        self.optim.zero_grad()

        return total_H / float(M), total_R / float(M), total_V / float(M)
