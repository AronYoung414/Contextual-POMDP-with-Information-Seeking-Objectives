import numpy as np
import random


class ContinuousLightDarkPOMDP:
    """
    Continuous Light–Dark POMDP with context-dependent light regions:

      Context 0:
          Light region = x > 0        → low variance observations
          Dark region  = x <= 0       → high variance observations

      Context 1:
          Light region = x < 0        → low variance observations
          Dark region  = x >= 0       → high variance observations

    All other mechanics identical to your original version:
      - continuous state
      - actions: left, right, observe
      - continuous noisy observations for 'o'
      - reward regions (goal/detector) per context
    """

    def __init__(self,
                 step_size=1.0,
                 process_noise=0.1,
                 light_noise=0.2,
                 dark_noise=4.0):
        # -------------------------------
        # CONTEXTS
        # -------------------------------
        self.contexts = [0, 1]
        self.context_distribution = {0: 0.5, 1: 0.5}

        # -------------------------------
        # ACTIONS
        # -------------------------------
        self.actions = ['l', 'r', 'o']
        self.action_size = len(self.actions)

        # -------------------------------
        # DYNAMICS
        # -------------------------------
        self.step_size = step_size
        self.process_noise = process_noise

        # -------------------------------
        # NOISE VARIANCES
        # -------------------------------
        self.light_noise = light_noise
        self.dark_noise = dark_noise

        # -------------------------------
        # INITIAL DISTRIBUTIONS
        # -------------------------------
        self.initial_dist = {
            0: (1.0, 0.01),    # mean, variance
            1: (-1.0, 0.01),
        }

        # -------------------------------
        # REWARD STRUCTURE
        # -------------------------------
        self.reward_func = {
            0: {  # context 0
                "goal_mu": 4.0,
                "goal_band": 1,
                "reward": 3,

                "det_mu": -3.0,
                "det_band": 1,
                "penalty": -2
            },
            1: {  # context 1
                "goal_mu": -4.0,
                "goal_band": 1,
                "reward": 3,

                "det_mu": 3.0,
                "det_band": 1,
                "penalty": -2
            }
        }

    # =========================================================
    # SEEDING
    # =========================================================
    def seed(self, sd):
        random.seed(sd)
        np.random.seed(sd)

    # =========================================================
    # INITIAL STATE
    # =========================================================
    def sample_initial_state(self, context):
        mu, var = self.initial_dist[context]
        return float(np.random.normal(mu, np.sqrt(var)))

    # =========================================================
    # STATE TRANSITION
    # =========================================================
    def next_state_sampler(self, x, action, context):
        """
        Continuous motion model:
          x' = x + step_direction + Gaussian(process_noise)
        """
        noise = np.random.normal(0, np.sqrt(self.process_noise))
        if action == 'l':
            dx = -self.step_size + noise
        elif action == 'r':
            dx = +self.step_size + noise
        else:  # 'o'
            dx = 0.0

        return float(x + dx)

    # =========================================================
    # OBSERVATION NOISE (LIGHT–DARK by CONTEXT)
    # =========================================================
    def _obs_sigma(self, x, context):
        """
        Context 0:
            light region = x > 0
            dark region  = x <= 0

        Context 1:
            light region = x < 0
            dark region  = x >= 0
        """
        if context == 0:
            if x > 0:
                return self.light_noise
            else:
                return self.dark_noise

        else:  # context == 1
            if x < 0:
                return self.light_noise
            else:
                return self.dark_noise

    # =========================================================
    # OBSERVATION FUNCTION
    # =========================================================
    def observation_sampler(self, x, action, context):
        """
        If action in ['l','r'], no observation (z=None).
        If action == 'o', z ~ N(x, σ(x,context)^2)
        """
        if action in ['l', 'r']:
            return None

        sigma = self._obs_sigma(x, context)
        return float(np.random.normal(x, sigma))

    # =========================================================
    # REWARD FUNCTION
    # =========================================================
    def reward_sampler(self, x, context):
        R = self.reward_func[context]

        # goal region
        if abs(x - R["goal_mu"]) <= R["goal_band"]:
            return R["reward"]

        # detector region
        if abs(x - R["det_mu"]) <= R["det_band"]:
            return random.choice([R["penalty"], 0])

        return 0

    # =========================================================
    # STEP FUNCTION
    # =========================================================
    def step(self, x, action, context):
        """
        Full environment step:
            - transition
            - observation
            - reward
        """
        x_next = self.next_state_sampler(x, action, context)
        obs = self.observation_sampler(x_next, action, context)
        rw = self.reward_sampler(x_next, context)
        return x_next, obs, rw
