import numpy as np
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from observable_operator_cp import LazyObservableOperatorCP


# ---------------------------------------------
# Utility
# ---------------------------------------------
def set_random_seeds(sd=42):
    np.random.seed(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    import random
    random.seed(sd)


# ---------------------------------------------
# Policy Network
# ---------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, obs_vocab_size, action_size, hidden_dim=64, max_seq_len=20):
        super().__init__()
        self.obs_vocab_size = obs_vocab_size
        self.action_size = action_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.obs_embedding = nn.Embedding(obs_vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_size)

    def forward(self, obs_sequences):
        embedded = self.obs_embedding(obs_sequences)
        _, (hidden, _) = self.lstm(embedded)
        final_hidden = hidden[-1]
        logits = self.action_head(final_hidden)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


# ------------------------------------------------
# Variational Policy Gradient (CPOMDP-correct)
# ------------------------------------------------
class VariationalPolicyGradientCP:
    """
    Implements algorithm A from your PDF:
        ∇θ L = E[(logPθ(y) − R/τ) ∇θ logPθ(y|c)]
    """

    def __init__(self, env, policy_net, tau, horizon, step_size):
        self.env = env
        self.policy_net = policy_net
        self.tau = tau
        self.T = horizon
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=step_size)

        # observable operators indexed by (context,obs,act)
        self.observable_operator = LazyObservableOperatorCP(env)

    # ------------------------------------------------
    # Policy probability πθ(a | obs-history)
    # ------------------------------------------------
    def pi_theta(self, obs_history, a_idx):
        if len(obs_history) == 0:
            obs_tensor = torch.tensor([[self.env.start_idx]], dtype=torch.long)
        else:
            idxs = [self.env.observations.index(o) for o in obs_history]
            obs_tensor = torch.tensor([idxs], dtype=torch.long)

        with torch.no_grad():
            probs, _ = self.policy_net(obs_tensor)
        return probs[0, a_idx].item()

    # ------------------------------------------------
    # Compute P(o0:T | a0:T, c) using observable operators
    # ------------------------------------------------
    def p_obs_given_actions(self, context, obs_list, act_list):
        mu0 = self.env.initial_dist[context]
        one_vec = np.ones((1, self.env.state_size))

        oo = self.observable_operator.get_operator(context, obs_list[-1], act_list[-1])
        probs = one_vec @ oo

        for t in reversed(range(len(obs_list) - 1)):
            oo = self.observable_operator.get_operator(context, obs_list[t], act_list[t])
            probs = probs @ oo

        return float(probs @ mu0)

    def p_obs0(self, context, obs0, act0):
        mu0 = self.env.initial_dist[context]
        oo = self.observable_operator.get_operator(context, obs0, act0)
        one_vec = np.ones((1, self.env.state_size))
        return float(one_vec @ oo @ mu0)

    # ------------------------------------------------
    # pθ(y | c)
    # ------------------------------------------------
    def p_y_given_c(self, context, obs_list, act_list):
        a_idx_list = [self.env.actions.index(act) for act in act_list]
        p_full = self.p_obs_given_actions(context, obs_list, act_list)
        p0 = self.p_obs0(context, obs_list[0], act_list[0])

        # policy product
        pi_prod = 1.0
        for t in range(len(a_idx_list)):
            pi_prod *= self.pi_theta(obs_list[:t], a_idx_list[t])

        return (p_full / (p0 + 1e-8)) * pi_prod

    # ------------------------------------------------
    # logPθ(y) and H(C|y)
    # ------------------------------------------------
    def compute_entropy_and_logp(self, obs_list, act_list):
        numerators = []

        for c in self.env.contexts:
            pyc = self.p_y_given_c(c, obs_list, act_list)
            numerators.append(pyc * self.env.context_distribution[c])

        Z = sum(numerators) + 1e-8
        logP = torch.log(torch.tensor(Z, dtype=torch.float32))

        # H(C|y)
        H = torch.zeros(1)
        for num in numerators:
            pcy = num / Z
            H += -pcy * torch.log(torch.tensor(pcy + 1e-8))
        return H, logP

    # ------------------------------------------------
    # ∇θ logPθ(y | c)
    # ------------------------------------------------
    def nabla_logP_y_given_c(self, context, obs_list, act_list):
        self.policy_net.zero_grad()
        a_idx_list = [self.env.actions.index(act) for act in act_list]
        log_terms = []

        # build log π term for each action
        for t in range(len(a_idx_list)):
            if t == 0:
                obs_idxs = [[self.env.start_idx]]
            else:
                obs_idxs = [[self.env.observations.index(o) for o in obs_list[:t]]]

            obs_tensor = torch.tensor(obs_idxs, dtype=torch.long)
            probs, _ = self.policy_net(obs_tensor)
            log_terms.append(torch.log(probs[0, a_idx_list[t]] + 1e-8))

        total_log = torch.stack(log_terms).sum()
        total_log.backward()

        grads = []
        for p in self.policy_net.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p))
            else:
                grads.append(p.grad.clone())
        return grads

    # ------------------------------------------------
    # SAMPLE A TRAJECTORY (this replaces your old sample_data)
    # ------------------------------------------------
    def sample_one_trajectory(self):
        context = np.random.choice(self.env.contexts)

        obs_list = []
        act_list = []
        st_list = []
        total_reward = 0

        # initial state
        st = self.env.initial_states[context][0]
        st_list.append(st)
        total_reward += self.env.reward_sampler(st, context)

        # first action
        act = self.sample_action([], context)
        act_list.append(act)

        # first observation
        obs = self.env.observation_function_sampler(st, act, context)
        obs_list.append(obs)

        # rollout
        for t in range(self.T - 1):
            st = self.env.next_state_sampler(st, act, context)
            st_list.append(st)
            total_reward += self.env.reward_sampler(st, context)

            act = self.sample_action(obs_list, context)
            act_list.append(act)

            obs = self.env.observation_function_sampler(st, act, context)
            obs_list.append(obs)

        return context, st_list, obs_list, act_list, total_reward

    # ------------------------------------------------
    # action sampler
    # ------------------------------------------------
    def sample_action(self, obs_list, context):
        if len(obs_list) == 0:
            obs_tensor = torch.tensor([[self.env.start_idx]], dtype=torch.long)
        else:
            idxs = [self.env.observations.index(o) for o in obs_list]
            obs_tensor = torch.tensor([idxs], dtype=torch.long)

        with torch.no_grad():
            probs, _ = self.policy_net(obs_tensor)
        act = np.random.choice(self.env.actions, p=probs[0].numpy())
        return act

    # ------------------------------------------------
    # ONE TRAINING ITERATION
    # ------------------------------------------------
    def train_step(self, M):
        total_entropy = 0
        total_reward = 0
        total_value = 0

        accumulated = None

        for _ in range(M):
            context, st_list, obs_list, act_list, reward = \
                self.sample_one_trajectory()

            H, logP = self.compute_entropy_and_logp(obs_list, act_list)

            grads = self.nabla_logP_y_given_c(context, obs_list, act_list)

            cst = logP - (reward / self.tau)

            total_entropy += H.item()
            total_reward += reward
            total_value += reward - self.tau * H.item()

            if accumulated is None:
                accumulated = [cst * g for g in grads]
            else:
                for i in range(len(grads)):
                    accumulated[i] += cst * grads[i]

        for i, p in enumerate(self.policy_net.parameters()):
            p.grad = accumulated[i] / M

        self.optimizer.step()
        self.optimizer.zero_grad()

        return (total_entropy / M,
                total_reward / M,
                total_value / M)
