import random
import numpy as np


class FunctionalContextEnv:
    """
    Functional environment with an added absorbing sink state.
    """

    def __init__(self, stoPar=0.1, obsNoise=0.1):
        self.contexts = [0, 1]
        self.context_distribution = {0: 0.5, 1: 0.5}

        self.stoPar = stoPar
        self.obsNoise = obsNoise

        # -----------------------------
        # Add sink state = 7
        # -----------------------------
        self.states = [0, 1, 2, 3, 4, 5, 6, 7]  # 7 is sink
        self.sink = 7

        self.actions = ['r', 'l', 's']
        self.observations = ['0', 'L', 'R', 'H']
        self.state_size = len(self.states)

        self.action_vocab = self.actions

    ############################################################
    # Context-dependent initial state
    ############################################################
    def get_initial_state_for_context(self, c):
        return 4 if c == 0 else 2

    def get_initial_distribution_for_context(self, c):
        mu = np.zeros((self.state_size, 1))
        s0 = self.get_initial_state_for_context(c)
        mu[self.states.index(s0), 0] = 1.0
        return mu

    ############################################################
    # Reward with sink behavior
    ############################################################
    def reward_sampler_for_context(self, s, c):
        if s == self.sink:
            return 0.0  # absorbing state reward = 0

        goal = self.get_goal_for_context(c)
        det = self.get_detector_for_context(c)

        if s == goal:
            return self.get_reward_value_for_context(c)

        if s == det:
            if c == 0:
                return random.choices([-2, 0], [0.5, 0.5])[0]
            else:
                return random.choices([-2, 0], [0.5, 0.5])[0]

        return 0.0

    ############################################################
    # Transition model with SINK
    ############################################################
    def get_transition_for_context(self, c):
        T = {s: {} for s in self.states}

        for s in self.states:
            for a in self.actions:
                T[s][a] = {}

                # Sink state is absorbing
                if s == self.sink:
                    T[s][a][self.sink] = 1.0
                    continue

                # If the agent is at the goal, next state is always sink
                if s == self.get_goal_for_context(c):
                    T[s][a][self.sink] = 1.0
                    continue

                # Normal transitions
                if a == 's':
                    T[s][a][s] = 1.0
                    continue

                target = s - 1 if a == 'l' else s + 1

                if target in self.states[:-1]:  # ignore sink index
                    T[s][a][target] = 1 - self.stoPar
                    T[s][a][s] = self.stoPar
                else:
                    T[s][a][s] = 1.0

        return T

    ############################################################
    # Emission model with SINK
    ############################################################
    def get_emission_for_context(self, c):
        E = {s: {} for s in self.states}
        det = self.get_detector_for_context(c)

        for s in self.states:
            for a in self.actions:
                E[s][a] = {}

                # Sink emits always '0'
                if s == self.sink:
                    E[s][a]['0'] = 1.0
                    continue

                if a in ['l', 'r']:
                    E[s][a]['0'] = 1.0
                    continue

                if s > det:
                    E[s][a]['L'] = 1 - self.obsNoise
                    E[s][a]['0'] = self.obsNoise
                elif s == det:
                    E[s][a]['H'] = 1 - self.obsNoise
                    E[s][a]['0'] = self.obsNoise
                else:
                    E[s][a]['R'] = 1 - self.obsNoise
                    E[s][a]['0'] = self.obsNoise

        return E

    ############################################################
    # Sampling
    ############################################################
    def next_state_sampler(self, s, a, c):
        T = self.get_transition_for_context(c)[s][a]
        return random.choices(list(T.keys()), list(T.values()))[0]

    def observation_sampler(self, s, a, c):
        E = self.get_emission_for_context(c)[s][a]
        return random.choices(list(E.keys()), list(E.values()))[0]

    ############################################################
    # Helpers
    ############################################################
    def get_goal_for_context(self, c):
        return 6 if c == 0 else 0

    def get_detector_for_context(self, c):
        return 5 if c == 0 else 1

    def get_reward_value_for_context(self, c):
        return 3 if c == 0 else 3
