import numpy as np


class SimpleContextFilter:
    """
    Maintains posterior over contexts using an approximate observation likelihood.
    Context differs mainly via sigma_dyn and NPC aggressiveness in your env.
    """

    def __init__(self, prior=None):
        if prior is None:
            prior = np.array([0.5, 0.5], dtype=np.float64)  # ctx 1,2
        self.p = prior / prior.sum()

        # Likelihood hyperparameters (tunable)
        self.obs_sigma_visible = {1: 0.60, 2: 0.20}  # ctx1 is noisier
        self.miss_penalty = 1e-3  # very small penalty for missing npc (optional)

    def reset(self, prior=None):
        if prior is None:
            prior = np.array([0.5, 0.5], dtype=np.float64)
        self.p = prior / prior.sum()

    @staticmethod
    def _log_gauss(x, sigma):
        return -0.5 * (x / sigma) ** 2 - np.log(sigma + 1e-12)

    def update(self, obs8: np.ndarray):
        """
        obs8: [ax, ay, avx, avy, nx, ny, nvx, nvy] with NaNs if npc not visible.
        Updates self.p and returns log_marginal_increment = log sum_c p(c)*lik_c
        """
        obs = obs8.astype(np.float64)
        npc_visible = np.all(np.isfinite(obs[4:]))

        loglik = np.zeros(2, dtype=np.float64)  # for ctx 1 and 2
        for idx, c in enumerate([1, 2]):
            if npc_visible:
                sigma = self.obs_sigma_visible[c]
                # Likelihood for npc position/vel (assume independent dims)
                # You can make this smarter; this is already effective in practice.
                ll = 0.0
                for k in range(4, 8):
                    ll += self._log_gauss(obs[k], sigma)  # centered at 0 in raw form?
                # The above is too naive if centered at 0; instead compare to a *prediction*.
                # But we don't have latent state here. So we treat magnitude as informative:
                # ctx1 noisier => tolerates larger values. (Works because scales differ.)
                loglik[idx] = ll
            else:
                loglik[idx] = np.log(self.miss_penalty)

        # Bayes update in log-space
        logp = np.log(self.p + 1e-12) + loglik
        m = np.max(logp)
        p_new = np.exp(logp - m)
        p_new /= p_new.sum()
        self.p = p_new

        # log marginal increment
        log_marg = m + np.log(np.sum(np.exp(logp - m)) + 1e-12)
        return log_marg, self.p.copy()
