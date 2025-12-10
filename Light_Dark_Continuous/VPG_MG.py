import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


# =====================================================================
#   BASIC GAUSSIAN UTILITIES
# =====================================================================

def normal_pdf(x, mu, var):
    return (1.0 / math.sqrt(2 * math.pi * var)) * math.exp(-(x - mu) ** 2 / (2 * var))


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# =====================================================================
#   TRUNCATED GAUSSIAN POSTERIOR
# =====================================================================

def truncated_gaussian_posterior(mu, var, region):
    std = math.sqrt(var)

    if region == 1:  # x > 0
        a = (0 - mu) / std
        Z = 1 - normal_cdf(a)
        if Z < 1e-12:
            return mu, var
        alpha = normal_pdf(a, 0, 1) / Z
        mu_new = mu + std * alpha
        var_new = var * (1 + a * alpha - alpha * alpha)
        return mu_new, var_new

    else:  # x < 0
        a = (0 - mu) / std
        Z = normal_cdf(a)
        if Z < 1e-12:
            return mu, var
        alpha = normal_pdf(a, 0, 1) / Z
        mu_new = mu - std * alpha
        var_new = var * (1 - a * alpha - alpha * alpha)
        return mu_new, var_new


# =====================================================================
#   REGION LIKELIHOOD
# =====================================================================

def region_likelihood(mu, var, z, sigma_r, region):
    tau2 = var + sigma_r * sigma_r
    m = (var * z + sigma_r * sigma_r * mu) / tau2
    v = (var * sigma_r * sigma_r) / tau2

    a = m / math.sqrt(v)

    if region == 1:
        mass = normal_cdf(a)
    else:
        mass = normal_cdf(-a)

    if mass < 1e-12:
        mass = 1e-12

    pred = normal_pdf(z, mu, tau2)
    return pred * mass


# =====================================================================
#   MIXTURE COLLAPSE TO 2 COMPONENTS
# =====================================================================

def collapse_to_two(components):
    wL = sum(c["w"] for c in components if c["region"] == 0)
    wR = sum(c["w"] for c in components if c["region"] == 1)

    if wL < 1e-12: wL = 1e-12
    if wR < 1e-12: wR = 1e-12

    muL = sum(c["w"] * c["mu"] for c in components if c["region"] == 0) / wL
    muR = sum(c["w"] * c["mu"] for c in components if c["region"] == 1) / wR

    varL = sum(c["w"] * (c["var"] + (c["mu"] - muL)**2)
               for c in components if c["region"] == 0) / wL
    varR = sum(c["w"] * (c["var"] + (c["mu"] - muR)**2)
               for c in components if c["region"] == 1) / wR

    return [
        {"w": wL, "mu": muL, "var": varL, "region": 0},
        {"w": wR, "mu": muR, "var": varR, "region": 1}
    ]


# =====================================================================
#   BELIEF PREDICTION + UPDATE
# =====================================================================

def belief_predict(components, step, proc_var):
    out = []
    for c in components:
        mu2 = c["mu"] + step
        out.append({
            "w": c["w"],
            "mu": mu2,
            "var": c["var"] + proc_var,
            "region": 0 if mu2 < 0 else 1
        })
    return out


def belief_update(components, z, region_noise_func):
    updated = []
    inc_total = 0.0

    for c in components:
        sigma_r = region_noise_func(c["region"])
        inc = region_likelihood(c["mu"], c["var"], z, sigma_r, c["region"])
        inc_total += c["w"] * inc

        mu2, var2 = truncated_gaussian_posterior(c["mu"], c["var"], c["region"])
        updated.append({
            "w": c["w"] * inc,
            "mu": mu2,
            "var": var2,
            "region": c["region"]
        })

    for c in updated:
        c["w"] /= inc_total

    collapsed = collapse_to_two(updated)

    return collapsed, inc_total


# =====================================================================
#   VPG CLASS
# =====================================================================

class VariationalPolicyGradient:
    def __init__(self, env, policy_net, tau, horizon, lr):
        self.env = env
        self.policy_net = policy_net
        self.tau = tau
        self.T = horizon
        self.opt = optim.Adam(policy_net.parameters(), lr=lr)

    # ---------------------------------------------------------
    # Build (T,2) continuous observation tensor for policy
    # ---------------------------------------------------------
    def make_policy_input(self, obs_hist):
        """
        Always return tensor of shape (1, T, 2).
        If obs_hist is empty, insert a dummy [0,0] row.
        """
        if len(obs_hist) == 0:
            arr = np.zeros((1, 1, 2), dtype=np.float32)  # (batch=1, seq=1, 2-dim)
        else:
            arr = np.array([obs_hist], dtype=np.float32)  # (1, T, 2)

        return torch.tensor(arr, dtype=torch.float32)

    # ---------------------------------------------------------
    # log π(a | obs_history)
    # ---------------------------------------------------------
    def logpi(self, obs_hist, a_idx):
        inp = self.make_policy_input(obs_hist)

        with torch.no_grad():
            probs, _ = self.policy_net(inp)

        p = probs[0, a_idx].item()
        return math.log(max(p, 1e-12))

    # ---------------------------------------------------------
    # P(y | c)
    # ---------------------------------------------------------
    def P_y_given_c(self, context, z_seq, m_seq, a_seq):
        mu0, var0 = self.env.initial_dist[context]
        components = [{
            "w": 1.0,
            "mu": mu0,
            "var": var0,
            "region": 0 if mu0 < 0 else 1
        }]

        logP_pi = 0.0
        likelihood = 1.0

        obs_hist = []

        for t in range(len(a_seq)):
            a = a_seq[t]
            a_idx = self.env.actions.index(a)

            logP_pi += self.logpi(obs_hist, a_idx)

            step = 0.
            if a == "l": step = -self.env.step_size
            if a == "r": step = +self.env.step_size

            components = belief_predict(components, step, self.env.process_noise)

            if m_seq[t] == 1:
                z = z_seq[t]
                components, inc = belief_update(
                    components,
                    z,
                    lambda region: (
                        self.env.light_noise if
                        ((context == 0 and region == 1) or (context == 1 and region == 0))
                        else self.env.dark_noise
                    )
                )
                likelihood *= inc
                obs_hist.append([z, 1.0])
            else:
                obs_hist.append([0.0, 0.0])

        return math.exp(logP_pi) * likelihood

    # ---------------------------------------------------------
    # P(y) and log P(y)
    # ---------------------------------------------------------
    def P_and_logP(self, z_seq, m_seq, a_seq):
        P = 0.0
        for c in self.env.contexts:
            P += self.env.context_distribution[c] * \
                 self.P_y_given_c(c, z_seq, m_seq, a_seq)
        return P, math.log(max(P, 1e-12))

    # ---------------------------------------------------------
    # Entropy H(C | y)
    # ---------------------------------------------------------
    def entropy(self, z_seq, m_seq, a_seq, P):
        unnorm = []
        for c in self.env.contexts:
            Pc = self.P_y_given_c(c, z_seq, m_seq, a_seq)
            unnorm.append(self.env.context_distribution[c] * Pc)

        Z = sum(unnorm)
        post = [u / Z for u in unnorm]

        H = 0.0
        for p in post:
            H -= p * math.log(max(p, 1e-12))
        return H

    # ---------------------------------------------------------
    # grad log π
    # ---------------------------------------------------------
    def grad_logpi(self, obs_hist, a_idx):
        self.policy_net.zero_grad()

        inp = self.make_policy_input(obs_hist)
        probs, _ = self.policy_net(inp)
        logp = torch.log(probs[0, a_idx] + 1e-12)
        logp.backward()

        grads = []
        for p in self.policy_net.parameters():
            grads.append(p.grad.clone() if p.grad is not None else torch.zeros_like(p))
        return grads

    # ---------------------------------------------------------
    # sample trajectory
    # ---------------------------------------------------------
    def sample_trajectory(self):
        context = np.random.choice(self.env.contexts)
        x = self.env.sample_initial_state(context)

        z_seq, m_seq, a_seq = [], [], []
        obs_hist = []
        reward = 0.0

        for t in range(self.T):
            a_idx, a = self.sample_action(obs_hist)
            a_seq.append(a)

            x, z, r = self.env.step(x, a, context)
            reward += r

            if z is not None:
                z_seq.append(z)
                m_seq.append(1)
                obs_hist.append([z, 1.0])
            else:
                z_seq.append(0.0)
                m_seq.append(0)
                obs_hist.append([0.0, 0.0])

        return context, z_seq, m_seq, a_seq, reward

    def sample_action(self, obs_hist):
        inp = self.make_policy_input(obs_hist)
        with torch.no_grad():
            probs, _ = self.policy_net(inp)
        p = probs[0].numpy()
        idx = np.random.choice(len(p), p=p)
        return idx, self.env.actions[idx]

    # ---------------------------------------------------------
    # training
    # ---------------------------------------------------------
    def train_step(self, M):
        total_H = 0.0
        total_R = 0.0
        total_V = 0.0

        accumulated = None

        for _ in range(M):
            _, z_seq, m_seq, a_seq, reward = self.sample_trajectory()

            P, logP = self.P_and_logP(z_seq, m_seq, a_seq)
            H = self.entropy(z_seq, m_seq, a_seq, P)
            V = reward - self.tau * H

            total_H += H
            total_R += reward
            total_V += V

            # gradients
            obs_hist = []
            grads_sum = None

            for t, a in enumerate(a_seq):
                a_idx = self.env.actions.index(a)
                g = self.grad_logpi(obs_hist, a_idx)

                if m_seq[t] == 1:
                    obs_hist.append([z_seq[t], 1.0])
                else:
                    obs_hist.append([0.0, 0.0])

                if grads_sum is None:
                    grads_sum = g
                else:
                    for i in range(len(g)):
                        grads_sum[i] += g[i]

            cst = logP - reward / self.tau

            if accumulated is None:
                accumulated = [cst * g for g in grads_sum]
            else:
                for i in range(len(accumulated)):
                    accumulated[i] += cst * grads_sum[i]

        for p, g in zip(self.policy_net.parameters(), accumulated):
            p.grad = g / M

        self.opt.step()
        self.opt.zero_grad()

        return total_H / M, total_R / M, total_V / M
