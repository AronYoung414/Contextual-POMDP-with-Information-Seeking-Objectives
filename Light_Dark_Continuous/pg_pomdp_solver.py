import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentPolicy(nn.Module):
    """
    History-based policy: consumes sequence of [z, m] and outputs action probs.
    """
    def __init__(self, obs_dim=2, hidden_dim=64, action_dim=3):
        super().__init__()
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.pi = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_seq, h=None):
        # obs_seq: (B, T, obs_dim)
        out, h = self.rnn(obs_seq, h)
        logits = self.pi(out[:, -1, :])  # last step
        probs = F.softmax(logits, dim=-1)
        return probs, h

class RecurrentValue(nn.Module):
    """
    Baseline V(h_t): same history encoder.
    """
    def __init__(self, obs_dim=2, hidden_dim=64):
        super().__init__()
        self.rnn = nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, obs_seq, h=None):
        out, h = self.rnn(obs_seq, h)
        value = self.v(out[:, -1, :]).squeeze(-1)  # (B,)
        return value, h

def sample_context(env, rng):
    cs = env.contexts
    ps = np.array([env.context_distribution[c] for c in cs], dtype=float)
    ps = ps / ps.sum()
    return int(rng.choice(cs, p=ps))

def rollout_episode(env, policy, value_net, horizon, rng, device="cpu"):
    """
    Roll one episode, collect trajectory for REINFORCE + baseline.
    """
    context = sample_context(env, rng)
    x = env.sample_initial_state(context)

    obs_hist = []   # list of [z, m]
    logps = []
    values = []
    rewards = []

    for t in range(horizon):
        if t == 0:
            obs_seq = torch.zeros((1, 1, 2), dtype=torch.float32, device=device)
        else:
            obs_seq = torch.tensor([obs_hist], dtype=torch.float32, device=device)

        probs, _ = policy(obs_seq)
        dist = torch.distributions.Categorical(probs=probs[0])
        a_idx = dist.sample()
        logp = dist.log_prob(a_idx)

        # map index -> action string in env.actions order
        a = env.actions[int(a_idx.item())]  # expects ['l','r','o']

        x, z, r = env.step(x, a, context)

        # encode observation: z is None unless 'o'
        if z is None:
            obs_hist.append([0.0, 0.0])
        else:
            obs_hist.append([float(z), 1.0])

        v, _ = value_net(obs_seq)
        logps.append(logp)
        values.append(v[0])
        rewards.append(float(r))

    # returns
    returns = np.zeros(horizon, dtype=np.float32)
    G = 0.0
    for t in reversed(range(horizon)):
        G = rewards[t] + G
        returns[t] = G

    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    logps_t = torch.stack(logps)
    values_t = torch.stack(values)

    total_return = float(sum(rewards))
    return total_return, logps_t, values_t, returns_t

def train_pg_pomdp(
    env,
    horizon=20,
    num_episodes=5000,
    lr=3e-4,
    hidden_dim=64,
    seed=0,
    device="cpu",
    log_every=50,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    policy = RecurrentPolicy(obs_dim=2, hidden_dim=hidden_dim, action_dim=len(env.actions)).to(device)
    value_net = RecurrentValue(obs_dim=2, hidden_dim=hidden_dim).to(device)

    optim_pi = torch.optim.Adam(policy.parameters(), lr=lr)
    optim_v  = torch.optim.Adam(value_net.parameters(), lr=lr)

    returns_log = []

    for ep in range(1, num_episodes + 1):
        env.seed(int(rng.integers(0, 10**9)))

        R, logps, values, returns_t = rollout_episode(env, policy, value_net, horizon, rng, device=device)
        returns_log.append(R)

        # Advantage
        adv = (returns_t - values.detach())

        # Policy loss: REINFORCE with baseline
        loss_pi = -(logps * adv).mean()

        # Value loss: regression
        loss_v = F.mse_loss(values, returns_t)

        optim_pi.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optim_pi.step()

        optim_v.zero_grad()
        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        optim_v.step()

        if ep % log_every == 0:
            avg = float(np.mean(returns_log[-log_every:]))
            print(f"[ep {ep:5d}] avg_return={avg:.3f}  loss_pi={loss_pi.item():.4f} loss_v={loss_v.item():.4f}")

    return policy, value_net, np.array(returns_log, dtype=np.float32)
