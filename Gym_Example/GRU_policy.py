import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Drop-in same signature as before.
    Converts env obs (8,) with NaNs into (16,) = [filled_obs(8), nan_mask(8)].
    """
    obs = obs.astype(np.float32)
    nan_mask = np.isnan(obs).astype(np.float32)
    obs_filled = np.nan_to_num(obs, nan=0.0).astype(np.float32)
    x = np.concatenate([obs_filled, nan_mask], axis=0)  # (16,)
    return torch.from_numpy(x).to(device)


class GRUGaussianPolicy(nn.Module):
    """
    History-dependent (recurrent) policy:
      z_t = GRU(z_{t-1}, [obs_t, mask_t, a_{t-1}])
      a_t ~ tanh(N(mu(z_t), std))

    Drop-in behavior:
      - call reset_hidden() at episode start
      - act(obs_tensor) returns action in (-1,1)
      - log_prob(obs_tensor, action_tensor) returns log pi(a_t | history up to t)
    """
    def __init__(
        self,
        obs_dim: int = 16,
        act_dim: int = 2,
        hidden: int = 128,
        gru_hidden: int = 128,
        log_std_init: float = -0.5,
        include_prev_action: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gru_hidden = gru_hidden
        self.include_prev_action = include_prev_action

        in_dim = obs_dim + (act_dim if include_prev_action else 0)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
        )

        self.gru = nn.GRUCell(hidden, gru_hidden)

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )

        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        # Persistent hidden state for a single environment rollout (not for batching).
        self._h = None
        self._prev_a = None  # in (-1,1)

    def reset_hidden(self, device: torch.device | None = None, batch_size: int = 1):
        """
        Call at episode reset. If you ever vectorize envs, set batch_size>1.
        """
        if device is None:
            device = next(self.parameters()).device
        self._h = torch.zeros(batch_size, self.gru_hidden, device=device)
        self._prev_a = torch.zeros(batch_size, self.act_dim, device=device)

    def _ensure_state(self, obs_tensor: torch.Tensor):
        """
        Make sure hidden state exists and matches obs batch shape.
        """
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        B = obs_tensor.shape[0]
        device = obs_tensor.device

        if self._h is None or self._h.shape[0] != B or self._h.device != device:
            self.reset_hidden(device=device, batch_size=B)

        return obs_tensor

    def _step_hidden(self, obs_tensor: torch.Tensor):
        """
        One recurrent update using (obs, prev_action).
        Updates internal hidden state self._h and returns it.
        """
        obs_tensor = self._ensure_state(obs_tensor)

        if self.include_prev_action:
            x = torch.cat([obs_tensor, self._prev_a], dim=-1)
        else:
            x = obs_tensor

        e = self.encoder(x)
        self._h = self.gru(e, self._h)
        return self._h

    def dist(self, obs_tensor: torch.Tensor) -> D.Normal:
        """
        Advances hidden state by 1 step and returns action distribution for time t.
        """
        h = self._step_hidden(obs_tensor)
        mean = self.head(h)  # (B, act_dim)
        std = torch.exp(self.log_std).clamp(1e-3, 10.0)  # (act_dim,)
        # Broadcast std to (B, act_dim)
        return D.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Samples an action in (-1,1) and updates prev_action.
        Returns shape (act_dim,) if obs_tensor is (obs_dim,)
        else (B, act_dim).
        """
        single = (obs_tensor.dim() == 1)
        d = self.dist(obs_tensor)  # updates hidden
        pre_tanh = d.sample()
        a = torch.tanh(pre_tanh)

        # store prev action (batched)
        if a.dim() == 1:
            self._prev_a = a.unsqueeze(0)
        else:
            self._prev_a = a

        return a.squeeze(0) if single else a

    def log_prob(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes log pi(a_t | history) for the provided action (already squashed in (-1,1)).
        IMPORTANT: This call also advances the hidden state by one step.
        So in rollout you should call either:
          - act(...) then log_prob(...) on the SAME timestep (would advance twice) -> DON'T
        Instead, during rollout do:
          - compute log_prob using this function while also sampling action yourself, OR
          - use act() to sample and store action, and compute log_prob using stored pre_tanh (more complex)
        The trainer I gave previously calls act() and then log_prob() => that would double-step.
        So you must update your rollout to use `sample_and_log_prob`.
        """
        # Use helper below for correct usage.
        raise RuntimeError(
            "Use sample_and_log_prob(obs_tensor) during rollout, or call log_prob_only(obs_tensor, action_tensor, h_prev)."
        )

    def sample_and_log_prob(self, obs_tensor: torch.Tensor):
        """
        Correct drop-in for rollout:
          a_t, logpi_t = policy.sample_and_log_prob(obs_t)

        - Advances hidden once
        - Samples action
        - Returns action in (-1,1) and the tanh-corrected logprob
        """
        single = (obs_tensor.dim() == 1)
        obs_tensor_b = obs_tensor.unsqueeze(0) if single else obs_tensor

        d = self.dist(obs_tensor_b)  # advances hidden
        pre_tanh = d.rsample()       # rsample for lower-variance gradients

        a = torch.tanh(pre_tanh)

        # tanh-squashed logprob with change-of-variables
        eps = 1e-6
        logp_gauss = d.log_prob(pre_tanh).sum(dim=-1)
        log_det = torch.log(1 - a * a + eps).sum(dim=-1)
        logp = logp_gauss - log_det

        # update prev action
        self._prev_a = a.detach()

        if single:
            return a.squeeze(0), logp.squeeze(0)
        return a, logp
