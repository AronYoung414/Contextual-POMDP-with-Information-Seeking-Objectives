import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from GRU_policy import GRUGaussianPolicy, obs_to_tensor
from Gym_Example.archive_code.context_filter import SimpleContextFilter


class VPGVariationalTrainer:
    def __init__(
        self,
        make_env_fn,
        device="cpu",
        tau=10.0,
        lr=3e-4,
        gamma=0.99,
        batch_episodes=16,
        max_steps=250,
        use_reward_to_go=False,
        entropy_bonus=0.0,
    ):
        """
        make_env_fn(ctx:int) -> env instance
        """
        self.make_env_fn = make_env_fn
        self.device = torch.device(device)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.batch_episodes = int(batch_episodes)
        self.max_steps = int(max_steps)
        self.use_reward_to_go = bool(use_reward_to_go)
        self.entropy_bonus = float(entropy_bonus)

        self.policy = GRUGaussianPolicy().to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

        # Baselines to reduce variance (moving averages)
        self.baseline_logp = 0.0
        self.baseline_R = 0.0
        self.baseline_beta = 0.95

    def _rollout_one(self, ctx: int):
        env = self.make_env_fn(ctx)
        obs, _ = env.reset()
        self.policy.reset_hidden()

        cf = SimpleContextFilter()
        cf.reset()

        logps = []
        rewards = []
        log_marginals = []

        for t in range(self.max_steps):
            obs_t = obs_to_tensor(obs, self.device)

            a, logp = self.policy.sample_and_log_prob(obs_t)

            action_np = a.detach().cpu().numpy().astype(np.float32)
            next_obs, r, done, trunc, _ = env.step(action_np)

            logps.append(logp)
            rewards.append(float(r))

        # Discounted returns
        rewards_np = np.array(rewards, dtype=np.float32)
        if self.use_reward_to_go:
            G = np.zeros_like(rewards_np)
            running = 0.0
            for i in reversed(range(len(rewards_np))):
                running = rewards_np[i] + self.gamma * running
                G[i] = running
            R_total = float(G[0])
        else:
            # total discounted return
            R_total = 0.0
            g = 1.0
            for r in rewards_np:
                R_total += g * r
                g *= self.gamma

        # policy log-likelihood of trajectory (sum log pi(a_t|o_t))
        logp_traj = torch.stack(logps).sum()

        # log P_theta(y) approx:
        # log P_theta(y) = sum log pi + log sum_c P(c)P(o|a,c)
        # Our context filter returns log marginal increments for the observation model proxy,
        # so add them (they don't depend on theta; they act like a constant baseline but help scaling).
        log_obs_marg = float(np.sum(log_marginals)) if len(log_marginals) > 0 else 0.0
        logP_y = logp_traj + torch.tensor(log_obs_marg, device=self.device)

        return {
            "logp_traj": logp_traj,
            "logP_y": logP_y,
            "R_total": R_total,
            "T": len(rewards),
        }

    def train_step(self):
        # Collect episodes, alternate contexts for balance
        batch = []
        for k in range(self.batch_episodes):
            ctx = 1 if (k % 2 == 0) else 2
            batch.append(self._rollout_one(ctx))

        logp_trajs = torch.stack([b["logp_traj"] for b in batch])
        logP_ys = torch.stack([b["logP_y"] for b in batch])
        R = torch.tensor([b["R_total"] for b in batch], device=self.device)

        # Moving baselines
        mean_logP = float(logP_ys.detach().mean().cpu().item())
        mean_R = float(R.detach().mean().cpu().item())
        self.baseline_logp = self.baseline_beta * self.baseline_logp + (1 - self.baseline_beta) * mean_logP
        self.baseline_R = self.baseline_beta * self.baseline_R + (1 - self.baseline_beta) * mean_R

        # Weight from your Eq. (13)/(14): (log P_theta(y) - R/τ)
        # We subtract baselines to reduce variance (does not change expectation).
        w = (logP_ys - self.baseline_logp) - (R - self.baseline_R) / self.tau

        # Surrogate loss: E[ w * ∇ log P_theta(y|c) ] implemented as
        # loss = mean( w.detach() * logp_traj )
        # (minimize loss => gradient descent on L)
        loss = (w.detach() * logp_trajs).mean()

        # Optional entropy bonus (encourages exploration)
        if self.entropy_bonus != 0.0:
            # crude proxy: encourage larger std (since tanh policy)
            loss = loss - self.entropy_bonus * self.policy.log_std.mean()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        stats = {
            "loss": float(loss.detach().cpu().item()),
            "R_mean": mean_R,
            "logP_y_mean": mean_logP,
            "traj_len_mean": float(np.mean([b["T"] for b in batch])),
        }
        return stats
