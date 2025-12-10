import random
import numpy as np


class FunctionalContextEnv:
    """
    Clean, stateless, context-functional rewrite of the user's original environment.
    """

    def __init__(self, stoPar=0.1, obsNoise=0.1):
        self.contexts = [0, 1]
        self.context_distribution = {0: 0.5, 1: 0.5}

        self.stoPar = stoPar
        self.obsNoise = obsNoise

        self.states = [0, 1, 2, 3, 4, 5, 6]
        self.actions = ['r', 'l', 's']
        self.observations = ['0', 'L', 'R', 'H']  # NO PAD
        self.state_size = len(self.states)

    #####################################################################
    # Context-dependent initial state / dist
    #####################################################################
    def get_initial_state_for_context(self, c):
        return 4 if c == 0 else 2

    def get_initial_distribution_for_context(self, c):
        mu = np.zeros((self.state_size, 1))
        s0 = self.get_initial_state_for_context(c)
        mu[self.states.index(s0), 0] = 1.0
        return mu

    #####################################################################
    # Rewards, detectors, penalty
    #####################################################################
    def get_goal_for_context(self, c):
        return 6 if c == 0 else 0

    def get_reward_value_for_context(self, c):
        return 0.3 if c == 0 else 0.5

    def get_detector_for_context(self, c):
        return 5 if c == 0 else 1

    def get_penalty_value_for_context(self, c):
        return -0.2 if c == 0 else -1

    def reward_sampler_for_context(self, s, c):
        goal = self.get_goal_for_context(c)
        detector = self.get_detector_for_context(c)
        rew = self.get_reward_value_for_context(c)
        pen = self.get_penalty_value_for_context(c)

        if s == goal:
            return rew
        elif s == detector:
            if c == 0:
                return random.choices([pen, 0], [0.5, 0.5])[0]
            else:
                return random.choices([pen, 0], [0.2, 0.8])[0]
        else:
            return 0.0

    #####################################################################
    # Transition and emission for context c
    #####################################################################
    def get_transition_for_context(self, c):
        trans = {}
        for s in self.states:
            trans[s] = {}
            for a in self.actions:
                trans[s][a] = {}

                if a == 's':
                    trans[s][a][s] = 1.0
                    continue

                target = s - 1 if a == 'l' else s + 1
                if target in self.states:
                    trans[s][a][target] = 1 - self.stoPar
                    trans[s][a][s] = self.stoPar
                else:
                    trans[s][a][s] = 1.0
        return trans

    def get_emission_for_context(self, c):
        emiss = {}
        detector = self.get_detector_for_context(c)

        for s in self.states:
            emiss[s] = {}
            for a in self.actions:
                emiss[s][a] = {}

                if a in ['l', 'r']:
                    emiss[s][a]['0'] = 1.0
                    continue

                if s > detector:
                    emiss[s][a]['L'] = 1 - self.obsNoise
                    emiss[s][a]['0'] = self.obsNoise
                elif s == detector:
                    emiss[s][a]['H'] = 1 - self.obsNoise
                    emiss[s][a]['0'] = self.obsNoise
                else:
                    emiss[s][a]['R'] = 1 - self.obsNoise
                    emiss[s][a]['0'] = self.obsNoise
        return emiss

    #####################################################################
    # Sampling (stateless)
    #####################################################################
    def next_state_sampler(self, s, a, c):
        trans = self.get_transition_for_context(c)[s][a]
        next_s = random.choices(list(trans.keys()), list(trans.values()), k=1)[0]
        return next_s

    def observation_sampler(self, s, a, c):
        emiss = self.get_emission_for_context(c)[s][a]
        obs = random.choices(list(emiss.keys()), list(emiss.values()), k=1)[0]
        return obs
