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
                 light_noise=1,
                 dark_noise=16.0):
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
            0: (0.0, 0.09),  # mean, variance
            1: (0.0, 0.09),
        }

        # -------------------------------
        # REWARD STRUCTURE
        # -------------------------------
        self.reward_func = {
            0: {  # context 0
                "goal_high": 1000,
                "goal_low": 1,
                "reward": 1,

                "det_high": 0,
                "det_low": -1000,
                "penalty": -1.3
            },
            1: {  # context 1
                "goal_high": -1,
                "goal_low": -1000,
                "reward": 1,

                "det_high": 1000,
                "det_low": 0,
                "penalty": -1.3
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
    def next_state_sampler(self, x, action):
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
    def obs_sigma(self, x, context):
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

        sigma = self.obs_sigma(x, context)
        return float(np.random.normal(x, np.sqrt(sigma)))

    # =========================================================
    # REWARD FUNCTION
    # =========================================================
    def reward_sampler(self, x, context):
        R = self.reward_func[context]

        # goal region
        if (x > R["goal_low"]) and (x < R["goal_high"]):
            return R["reward"]

        # detector region
        if (x > R["det_low"]) and (x < R["det_high"]):
            return R["penalty"]

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
        x_next = self.next_state_sampler(x, action)
        obs = self.observation_sampler(x_next, action, context)
        rw = self.reward_sampler(x_next, context)
        return x_next, obs, rw
