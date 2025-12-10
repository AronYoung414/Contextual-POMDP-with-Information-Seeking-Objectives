#!/usr/bin/env python3
# ================================================================
#   Contextual POMDP Variational Policy Gradient (Single File)
#   ENV–B with Sink State + LSTM Policy + Variational PG
#   Author: ChatGPT (customized for Chongyang)
# ================================================================

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import random
import matplotlib.pyplot as plt
import os


# ================================================================
#  Utility: set random seeds
# ================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================================================================
#  ENV: Contextual Environment with SINK state
# ================================================================

from line_grid_sinked import FunctionalContextEnv


# ================================================================
#   Policy Network (LSTM)
# ================================================================
class PolicyNetwork(nn.Module):
    def __init__(self, obs_vocab, actions, hidden_dim=64):
        super().__init__()
        self.obs_vocab = obs_vocab
        self.actions = actions

        self.embed = nn.Embedding(len(obs_vocab), hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, len(actions))

    def forward(self, obs_seq_idx):
        """
        obs_seq_idx : shape (1, T)
        """
        emb = self.embed(obs_seq_idx)
        out, _ = self.lstm(emb)
        logits = self.out(out[:, -1, :])
        probs = torch.softmax(logits, dim=-1)
        return probs, logits


# ================================================================
# Variational Policy Gradient Loss (Eq. 13 & Eq. 18)
# ================================================================
def vpg_loss(env, policy, obs_seq, act_seq, context, tau):
    # ---------------------------
    # 1. log P_theta(y|c)
    # ---------------------------
    logp_y_given_c = 0
    for t in range(len(act_seq)):
        prefix = obs_seq[:t + 1]
        obs_idx = torch.tensor([[env.observations.index(o) for o in prefix]])
        probs, _ = policy(obs_idx)
        dist = Categorical(probs)
        a_idx = env.actions.index(act_seq[t])
        logp_y_given_c = logp_y_given_c + dist.log_prob(
            torch.tensor([a_idx])
        ).squeeze()

    # ---------------------------
    # 2. Compute weight, no grad
    # ---------------------------
    with torch.no_grad():

        # log P_theta(y)
        logP_list = []
        for c in env.contexts:
            lp = 0
            for t in range(len(act_seq)):
                prefix = obs_seq[:t + 1]
                obs_idx = torch.tensor([[env.observations.index(o) for o in prefix]])
                probs, _ = policy(obs_idx)
                dist = Categorical(probs)
                a_idx = env.actions.index(act_seq[t])
                lp = lp + dist.log_prob(torch.tensor([a_idx])).squeeze()
            logP_list.append(lp + np.log(env.context_distribution[c]))

        logP_y = torch.logsumexp(torch.stack(logP_list), dim=0)

        # R_c(y)
        Rc = 0.0
        s = env.get_initial_state_for_context(context)
        for a in act_seq:
            Rc += env.reward_sampler_for_context(s, context)
            s = env.next_state_sampler(s, a, context)
        # print(logP_y)
        # print(Rc/tau)

        weight = logP_y - Rc / tau

    # ----------- CRITICAL -----------
    weight = weight.detach()
    # --------------------------------

    # ---------------------------
    # 3. Final VPG loss
    # ---------------------------
    return weight * logp_y_given_c


# ================================================================
#   Trajectory Sampling
# ================================================================
def sample_trajectory(env, policy, horizon, context):
    """
    Returns:
        states, obs_seq, act_seq, total_reward
    """
    states = []
    obs_seq = []
    act_seq = []

    s = env.get_initial_state_for_context(context)
    states.append(s)

    # Start with dummy observation "0"
    o = '0'
    obs_seq.append(o)

    total_reward = 0.0

    for t in range(horizon):

        # Policy input: full prefix
        obs_idx = torch.tensor([[env.observations.index(o) for o in obs_seq]])
        probs, _ = policy(obs_idx)
        dist = Categorical(probs)
        a_idx = dist.sample().item()
        a = env.actions[a_idx]
        act_seq.append(a)

        total_reward += env.reward_sampler_for_context(s, context)
        o = env.observation_sampler(s, a, context)
        obs_seq.append(o)

        s = env.next_state_sampler(s, a, context)
        states.append(s)

        if s == env.sink:
            break

    return states, obs_seq, act_seq, total_reward


# ================================================================
# Entropy H(C | y, a)
# ================================================================
def entropy_C(env, obs_seq, act_seq):
    # Forward recursion
    def p_y_given_c(c):
        S = env.states
        idx = {s: i for i, s in enumerate(S)}
        mu = torch.zeros(len(S))
        mu[idx[env.get_initial_state_for_context(c)]] = 1.0
        T = env.get_transition_for_context(c)
        E = env.get_emission_for_context(c)

        for t in range(len(act_seq)):
            o = obs_seq[t]
            a = act_seq[t]
            new_mu = torch.zeros_like(mu)
            for s_prev in S:
                for s_next, prob in T[s_prev][a].items():
                    emiss = E[s_next][a].get(o, 0.0)
                    new_mu[idx[s_next]] += mu[idx[s_prev]] * prob * emiss
            mu = new_mu
        return mu.sum()

    pcs = []
    for c in env.contexts:
        pcs.append(p_y_given_c(c) * env.context_distribution[c])

    pcs = torch.stack(pcs)
    Z = pcs.sum()
    if Z <= 1e-12:
        return torch.tensor(0.0)
    post = pcs / Z
    return -(post * torch.log2(post + 1e-12)).sum()


# ================================================================
#   Trainer
# ================================================================
class Trainer:
    def __init__(self, env, policy, horizon=15, lr=1e-3, tau=1.0, save_dir="results"):
        self.env = env
        self.policy = policy
        self.horizon = horizon
        self.tau = tau

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.loss_curve = []
        self.reward_curve = []
        self.entropy_curve = []

    def train(self, iterations=200, batch=20):

        for it in range(iterations):
            batch_loss = 0.0
            batch_R = 0.0
            batch_H = 0.0

            for _ in range(batch):
                c = random.choice(self.env.contexts)
                states, obs_seq, act_seq, R = sample_trajectory(self.env, self.policy, self.horizon, c)
                Hc = entropy_C(self.env, obs_seq, act_seq)

                loss = vpg_loss(self.env, self.policy, obs_seq, act_seq, c, self.tau)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                batch_R += R
                batch_H += Hc.item()
                batch_loss += self.tau * Hc.item() - R

            self.loss_curve.append(batch_loss / batch)
            self.reward_curve.append(batch_R / batch)
            self.entropy_curve.append(batch_H / batch)

            print(f"[{it}] R={batch_R / batch:.3f}, H={batch_H / batch:.3f}")

        self.plot_curves()
        self.save_policy()

    # ------------------------------------------------------------
    def plot_curves(self):
        it = np.arange(len(self.loss_curve))
        plt.figure()
        plt.plot(it, self.reward_curve)
        plt.title("Reward")
        plt.savefig(f"{self.save_dir}/reward.png")
        plt.show()

        plt.figure()
        plt.plot(it, self.entropy_curve)
        plt.title("Entropy")
        plt.savefig(f"{self.save_dir}/entropy.png")
        plt.show()

        plt.figure()
        plt.plot(it, self.loss_curve)
        plt.title("Loss")
        plt.savefig(f"{self.save_dir}/loss.png")
        plt.show()

        print("Saved training plots.")

    # ------------------------------------------------------------
    def save_policy(self):
        path = f"{self.save_dir}/policy.pt"
        torch.save(self.policy.state_dict(), path)
        print(f"Saved policy to {path}")


# ================================================================
# Evaluation
# ================================================================
def load_policy(env, path, hidden_dim=64):
    policy = PolicyNetwork(env.observations, env.actions, hidden_dim)
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


def evaluate(env, policy, horizon=15, rollouts=20):
    R_list, H_list = [], []
    for _ in range(rollouts):
        c = random.choice(env.contexts)
        states, obs_seq, act_seq, R = sample_trajectory(env, policy, horizon, c)
        H = entropy_C(env, obs_seq, act_seq)
        R_list.append(R)
        H_list.append(H.item())
        print("Trajectory:", states, obs_seq, act_seq, "R=", R, "H=", H.item())

    print("\nAvg Reward:", np.mean(R_list))
    print("Avg Entropy:", np.mean(H_list))


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    set_seed(1337)

    env = FunctionalContextEnv(stoPar=0.1, obsNoise=0.1)
    policy = PolicyNetwork(env.observations, env.actions, hidden_dim=64)

    trainer = Trainer(env, policy, horizon=10, lr=1e-3, tau=0.1, save_dir="../data_line_grid_3/exp_3")
    trainer.train(iterations=200, batch=20)

    print("\n=== Evaluation of learned policy ===")
    trained_policy = load_policy(env, "../data_line_grid_3/exp_2/policy.pt")
    evaluate(env, trained_policy, horizon=15, rollouts=30)
