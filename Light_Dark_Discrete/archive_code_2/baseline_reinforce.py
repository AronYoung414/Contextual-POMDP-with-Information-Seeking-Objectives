###############################################################
#  FULL INTEGRATED VARIATIONAL POLICY GRADIENT PROJECT FILE
#
#  - Functional context-dependent environment (no mutation)
#  - Stable LSTM policy
#  - Trajectory sampler
#  - Model-based entropy H(C | y,a)
#  - Correct REINFORCE objective:
#        J = E[ R - H(C|y,a) ]
#  - Batch training with PyTorch
###############################################################

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os


###############################################################
#  0. GLOBAL SEED
###############################################################

def set_seed(sd=1337):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)


###############################################################
#  1. FUNCTIONAL ENVIRONMENT (NO MUTATION)
###############################################################

from line_grid_sinked import FunctionalContextEnv

###############################################################
#  2. POLICY NETWORK (STABLE)
###############################################################

class PolicyNetwork(nn.Module):
    def __init__(self, obs_vocab, action_vocab, hidden_dim=64):
        super().__init__()
        self.obs_vocab = obs_vocab
        self.action_vocab = action_vocab

        self.embedding = nn.Embedding(len(obs_vocab), hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, len(action_vocab))

    def forward(self, obs_idx_seq):
        """
        obs_idx_seq: (1, T)
        """
        emb = self.embedding(obs_idx_seq)
        _, (h_last, _) = self.lstm(emb)
        logits = self.head(h_last.squeeze(0))
        probs = F.softmax(logits, dim=-1)
        return probs, logits


###############################################################
#  3. TRAJECTORY SAMPLER
###############################################################

def sample_trajectory(env, policy, horizon, context):
    obs_seq = []
    act_seq = []
    total_reward = 0.0

    # initial state
    s = env.get_initial_state_for_context(context)

    # initial dummy observation
    o = '0'
    obs_seq.append(o)

    for t in range(horizon):
        # prepare policy input
        obs_indices = torch.tensor(
            [[env.observations.index(o) for o in obs_seq]],
            dtype=torch.long
        )

        probs, _ = policy(obs_indices)
        dist = Categorical(probs)
        a_idx = dist.sample()
        a = env.actions[a_idx]

        act_seq.append(a)

        # reward
        r = env.reward_sampler_for_context(s, context)
        total_reward += r

        # next observation
        o = env.observation_sampler(s, a, context)
        obs_seq.append(o)

        # next state
        s = env.next_state_sampler(s, a, context)

    return obs_seq, act_seq, total_reward


###############################################################
#  4. Compute p(y, a | c) and H(C | y, a)
###############################################################

def compute_p_y_a_c(env, obs_seq, act_seq, c):
    T = len(act_seq)
    S = env.states
    idx = {s: i for i, s in enumerate(S)}

    mu = torch.tensor(env.get_initial_distribution_for_context(c), dtype=torch.float32)
    mu = mu.squeeze(1)  # shape (S,)

    trans = env.get_transition_for_context(c)
    emiss = env.get_emission_for_context(c)

    for t in range(T):
        a = act_seq[t]
        o = obs_seq[t]

        new_mu = torch.zeros_like(mu)
        for s_prev in S:
            i_prev = idx[s_prev]

            # emission
            e = emiss[s_prev][a].get(o, 0.0)

            for s_next in trans[s_prev][a]:
                i_next = idx[s_next]
                new_mu[i_next] += mu[i_prev] * trans[s_prev][a][s_next] * e

        mu = new_mu

    return torch.sum(mu)


def compute_context_entropy(env, obs_seq, act_seq):
    p_list = []
    for c in env.contexts:
        pyac = compute_p_y_a_c(env, obs_seq, act_seq, c)
        prior = env.context_distribution[c]
        p_list.append(pyac * prior)

    p_y_a = torch.sum(torch.stack(p_list))
    if p_y_a <= 1e-12:
        return torch.tensor(0.0)
    post = torch.stack(p_list) / p_y_a
    # print(obs_seq)
    entropy = -torch.sum(post * torch.log2(post + 1e-12))
    return entropy


###############################################################
#  5. REINFORCE LOSS
###############################################################

def reinforce_loss(policy, env, obs_seq, act_seq, R, entropy, tau):
    logp_sum = torch.tensor(0.0)

    for t in range(len(act_seq)):
        obs_indices = torch.tensor(
            [[env.observations.index(o) for o in obs_seq[:t + 1]]],
            dtype=torch.long
        )
        probs, _ = policy(obs_indices)
        dist = Categorical(probs)

        a_idx = env.actions.index(act_seq[t])

        # --- THE FINAL ALWAYS-CORRECT VERSION ---
        a_tensor = torch.tensor([a_idx], dtype=torch.long)  # shape = (1,)
        logp_sum = logp_sum + dist.log_prob(a_tensor).sum()

    advantage = R - tau * entropy
    return -advantage * logp_sum

    ###############################################################
    #  6. TRAINER
    ###############################################################


class Trainer:
    def __init__(self, env, policy, horizon=10, lr=1e-3, save_dir="results"):
        self.env = env
        self.policy = policy
        self.horizon = horizon
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

        env.action_vocab = env.actions

        # Create save directory
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Training curves
        self.rewards_curve = []
        self.entropy_curve = []
        self.loss_curve = []

    ###########################################################
    # TRAIN
    ###########################################################
    def train(self, iterations=200, batch_size=50, tau=1):
        for it in range(iterations):
            total_loss = 0.0
            avg_R = 0.0
            avg_H = 0.0

            for _ in range(batch_size):
                c = random.choice(self.env.contexts)

                obs_seq, act_seq, R = sample_trajectory(self.env, self.policy, self.horizon, c)
                Hc = compute_context_entropy(self.env, obs_seq, act_seq)
                loss = reinforce_loss(self.policy, self.env, obs_seq, act_seq, R, Hc, tau)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.detach().item()
                avg_R += R
                avg_H += Hc.detach().item()

            avg_loss = total_loss / batch_size
            avg_reward = avg_R / batch_size
            avg_entropy = avg_H / batch_size

            self.loss_curve.append(avg_loss)
            self.rewards_curve.append(avg_reward)
            self.entropy_curve.append(avg_entropy)

            print(f"[Iter {it}]  Loss={avg_loss:.3f}   Reward={avg_reward:.3f}   Entropy={avg_entropy:.3f}")

        # Save curves & policy
        self.save_curves()
        self.save_policy()
        self.plot_curves()

        return self.rewards_curve, self.entropy_curve, self.loss_curve

    ###########################################################
    # SAVE CURVES
    ###########################################################
    def save_curves(self):
        import numpy as np

        np.save(f"{self.save_dir}/reward_curve.npy", np.array(self.rewards_curve))
        np.save(f"{self.save_dir}/entropy_curve.npy", np.array(self.entropy_curve))
        np.save(f"{self.save_dir}/loss_curve.npy", np.array(self.loss_curve))

        print(f"Saved arrays to {self.save_dir}/[reward|entropy|loss]_curve.npy")

    ###########################################################
    # SAVE POLICY
    ###########################################################
    def save_policy(self):
        torch.save(self.policy.state_dict(), f"{self.save_dir}/policy_final.pt")
        print(f"Saved policy to {self.save_dir}/policy_final.pt")

    ###########################################################
    # PLOT CURVES
    ###########################################################
    def plot_curves(self):
        it = list(range(len(self.loss_curve)))

        # Reward plot
        plt.figure()
        plt.plot(it, self.rewards_curve)
        plt.title("Average Reward vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/reward_curve.png")

        # Entropy plot
        plt.figure()
        plt.plot(it, self.entropy_curve)
        plt.title("Entropy vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/entropy_curve.png")

        # Loss plot
        plt.figure()
        plt.plot(it, self.loss_curve)
        plt.title("Loss vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/loss_curve.png")

        print(f"Saved plots to {self.save_dir}/reward_curve.png, entropy_curve.png, loss_curve.png")


###############################################################
#  7. MAIN
###############################################################

if __name__ == "__main__":
    set_seed(1337)

    env = FunctionalContextEnv(stoPar=0.1, obsNoise=0.1)
    policy = PolicyNetwork(env.observations, env.actions, hidden_dim=64)

    trainer = Trainer(env=env, policy=policy, horizon=10, lr=1e-3, save_dir="../data_line_grid_2/exp_6_balanced")
    trainer.train(iterations=200, batch_size=200, tau=1)
