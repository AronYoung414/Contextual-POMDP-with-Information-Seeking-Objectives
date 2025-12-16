import numpy as np
import math
import torch


def gaussian_likelihood(z, mean, sigma):
    """
    Normal Gaussian likelihood N(z | mean, sigma^2)
    """
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * \
        math.exp(-(z - mean) ** 2 / (2 * sigma * sigma))


class ParticleFilter:
    """
    Particle filter that computes:
        - probability P_theta(y|c)
        - log probability logP = log(P)
    WITHOUT using log weights internally.

    Uses:  P = Π_t Z_t     where Z_t is observation normalization constant.
    """

    def __init__(self, env, policy, num_particles=2000):
        self.env = env
        self.policy = policy
        self.N = num_particles

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    def init_particles(self, context):
        mu0, var0 = self.env.initial_dist[context]
        particles = np.random.normal(mu0, math.sqrt(var0), self.N)
        weights = np.ones(self.N) / self.N
        return particles, weights

    # --------------------------------------------------------
    # Resampling
    # --------------------------------------------------------
    def safe_normalize(self, weights):
        """Normalize weights; handle NaN, Inf, zero-sum cases."""
        weights = np.asarray(weights, dtype=float)

        if np.any(np.isnan(weights)) or np.any(weights < 0):
            # Reset to uniform
            return np.ones_like(weights) / len(weights)

        s = np.sum(weights)
        if s <= 0 or np.isinf(s) or np.isnan(s):
            return np.ones_like(weights) / len(weights)

        return weights / s

    def resample(self, particles, weights):
        """Resample always with SAFE probability normalization."""
        weights = self.safe_normalize(weights)
        idx = np.random.choice(self.N, size=self.N, p=weights)
        new_particles = particles[idx]
        new_weights = np.ones(self.N) / self.N
        return new_particles, new_weights
    # --------------------------------------------------------
    # MAIN PF (Probability first, Log second)
    # --------------------------------------------------------
    def compute_trajectory_prob(self, context, z_seq, m_seq, act_seq):
        """
        Returns:
            (P, logP)
            where P is true probability of y under context c
        """

        particles, weights = self.init_particles(context)

        total_prob = 1.0  # P = Π Z_t

        obs_history = []
        import torch

        for t, (z, m, a) in enumerate(zip(z_seq, m_seq, act_seq)):

            # ---------------------------------------------
            # 1. Policy probability (in probability space)
            # ---------------------------------------------
            if t == 0:
                inp = torch.zeros((1, 1, 2))
            else:
                inp = torch.tensor([obs_history], dtype=torch.float32)

            probs, _ = self.policy(inp)
            a_idx = self.env.actions.index(a)
            pi_prob = float(probs[0, a_idx])

            # multiply into total probability
            total_prob *= pi_prob

            # ---------------------------------------------
            # 2. Propagate particles
            # ---------------------------------------------
            new_particles = np.zeros_like(particles)
            for i, x in enumerate(particles):
                new_particles[i] = self.env.next_state_sampler(x, a, context)
            particles = new_particles

            # ---------------------------------------------
            # 3. If sensing, apply observation update
            # ---------------------------------------------
            # If sensing event: likelihood update
            if m == 1 and a == 'o':
                # multiply weights by likelihood
                for i, x in enumerate(particles):
                    sigma = self.env.obs_sigma(x)
                    weights[i] *= gaussian_likelihood(z, x, sigma)

                # normalizing constant
                Z_t = np.sum(weights) + 1e-12
                total_prob *= Z_t

                # normalize for resampling
                weights = self.safe_normalize(weights)

                # resample
                particles, weights = self.resample(particles, weights)

            else:
                # movement step: STILL need normalized weights!
                weights = self.safe_normalize(weights)

            # ---------------------------------------------
            # 4. record observation
            # ---------------------------------------------
            obs_history.append([z, m])

        # return both
        logP = math.log(total_prob + 1e-300)
        return total_prob, logP
