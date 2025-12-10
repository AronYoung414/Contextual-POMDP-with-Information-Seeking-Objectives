import random
from random import choices

import numpy as np


class Environment:

    def __init__(self, context, stoPar, obsNoise):
        # context parameter
        self.context = context
        self.contexts = [0, 1]
        self.context_distribution = {0: 0.5, 1: 0.5}
        # parameter which controls environment noise
        self.stoPar = stoPar
        # parameter which controls observation noise
        self.obsNoise = obsNoise
        # Define states
        self.sink_state = 7
        self.states = [0, 1, 2, 3, 4, 5, 6, self.sink_state]
        # self.states = [0, 1, 2, 3, 4, 5, 6]
        self.state_size = len(self.states)
        # Define actions
        self.actions = ['r', 'l', 's']  # moving right, left and stay here
        self.action_size = len(self.actions)
        # Define initial state
        self.initial_states = self.get_initial_state()
        self.initial_dist = self.get_initial_distribution()
        # Define the observations
        self.observations = ['0', 'L', 'R', 'H']  # null, left, right, here observations, and dummy
        self.start_idx = 0  # use '0' as default first observation
        # Different settings depends on context
        self.rewards, self.goal = self.get_reward()
        self.penalty, self.detector = self.get_penalty()
        # Get transitions
        self.transition = self.get_transition()
        self.check_trans()
        # Get emission function
        self.emiss = self.get_emission_function()
        self.check_emission_function()

    def seed(self, sd):
        random.seed(sd)
        np.random.seed(sd)

    def get_initial_state(self):
        if self.context == 0:
            return [4]
        elif self.context == 1:
            return [2]
        else:
            raise ValueError('Invalid context parameter.')

    def get_initial_distribution(self):
        mu_0 = np.zeros([self.state_size, 1])
        for initial_st in self.initial_states:
            s_0 = self.states.index(initial_st)
            mu_0[s_0, 0] = 1 / len(self.initial_states)
        return mu_0

    def get_reward(self):
        if self.context == 0:
            return 0.3, 6
        elif self.context == 1:
            return 0.5, 0
        else:
            raise ValueError('Invalid context parameter.')

    def get_penalty(self):
        if self.context == 0:
            return -0.2, 5
        elif self.context == 1:
            return -1, 1
        else:
            raise ValueError('Invalid context parameter.')

    def check_inside(self, st):
        # If the state is valid or not
        if st in self.states:
            return True
        return False

    # def get_transition(self):
    #     trans = {}
    #     for st in self.states:
    #         trans[st] = {}
    #         for act in self.actions:
    #             if act == 's':
    #                 trans[st][act] = {}
    #                 trans[st][act][st] = 1  # stay there
    #             else:
    #                 trans[st][act] = {}
    #                 trans[st][act][st] = 0
    #                 if act == 'l':
    #                     tempst = st - 1  # moving to the left
    #                 elif act == 'r':
    #                     tempst = st + 1  # moving to the right
    #                 else:
    #                     raise ValueError('Invalid action.')
    #                 if self.check_inside(tempst):
    #                     trans[st][act][tempst] = 1 - self.stoPar
    #                     trans[st][act][st] = self.stoPar
    #                 else:
    #                     trans[st][act][st] = 1
    #     return trans

    def get_transition(self):
        trans = {}
        for st in self.states:
            trans[st] = {}
            for act in self.actions:

                # --- SINK STATE: always stay ---
                if st == self.sink_state:
                    trans[st][act] = {self.sink_state: 1.0}
                    continue

                # --- GOAL: transition to sink ---
                if st == self.goal:
                    trans[st][act] = {self.sink_state: 1.0}
                    continue

                # --- NORMAL transitions ---
                trans[st][act] = {}

                if act == 's':
                    trans[st][act][st] = 1.0

                else:
                    trans[st][act][st] = 0
                    if act == 'l':
                        tempst = st - 1
                    elif act == 'r':
                        tempst = st + 1
                    else:
                        raise ValueError("Invalid action")

                    if self.check_inside(tempst):
                        trans[st][act][tempst] = 1 - self.stoPar
                        trans[st][act][st] = self.stoPar
                    else:
                        trans[st][act][st] = 1.0
        return trans

    def check_trans(self):
        # Check if the transitions are constructed correctly
        for st in self.transition.keys():
            for act in self.transition[st].keys():
                if abs(sum(self.transition[st][act].values()) - 1) > 0.01:
                    print("st is:", st, "act is:", act, "sum is:", sum(self.transition[st][act].values()))
                    return False
        print("Transition is correct")
        return True

    # def get_emission_function(self):
    #     emiss = {}
    #     for st in self.states:
    #         emiss[st] = {}
    #         for act in self.actions:
    #             emiss[st][act] = {}
    #             if act == 's':  # start sensing
    #                 if st > self.detector:
    #                     emiss[st][act]['L'] = 1 - self.obsNoise
    #                     emiss[st][act]['0'] = self.obsNoise
    #                 elif st == self.detector:
    #                     emiss[st][act]['H'] = 1 - self.obsNoise
    #                     emiss[st][act]['0'] = self.obsNoise
    #                 elif st < self.detector:
    #                     emiss[st][act]['R'] = 1 - self.obsNoise
    #                     emiss[st][act]['0'] = self.obsNoise
    #                 else:
    #                     raise ValueError('Invalid state, action, or observation.')
    #             else:  # action is l or r
    #                 emiss[st][act]['0'] = 1
    #     return emiss

    def get_emission_function(self):
        emiss = {}
        for st in self.states:
            emiss[st] = {}
            for act in self.actions:

                # --- Sink emits dummy observation ---
                if st == self.sink_state:
                    emiss[st][act] = {'0': 1.0}
                    continue

                emiss[st][act] = {}
                if act == 's':
                    if st > self.detector:
                        emiss[st][act]['L'] = 1 - self.obsNoise
                        emiss[st][act]['0'] = self.obsNoise
                    elif st == self.detector:
                        emiss[st][act]['H'] = 1 - self.obsNoise
                        emiss[st][act]['0'] = self.obsNoise
                    elif st < self.detector:
                        emiss[st][act]['R'] = 1 - self.obsNoise
                        emiss[st][act]['0'] = self.obsNoise
                    else:
                        raise ValueError("Invalid state")
                else:
                    emiss[st][act]['0'] = 1.0
        return emiss

    def check_emission_function(self):
        for st in self.states:
            for act in self.actions:
                prob = 0
                observation_set = list(self.emiss[st][act].keys())
                for obs in observation_set:
                    prob += self.emiss[st][act][obs]
                if abs(prob - 1) > 0.01:
                    print(f"The emission is invalid.", self.emiss[st][act])
                    return False
        print("The check of emission function is done (POMDP).")
        return True

    def next_state_sampler(self, st, act):
        next_supp = list(self.transition[st][act].keys())
        next_prob = [self.transition[st][act][st_prime] for st_prime in next_supp]
        next_state = choices(next_supp, next_prob, k=1)[0]
        return next_state

    def observation_function_sampler(self, st, act):
        observation_set = list(self.emiss[st][act].keys())
        observation_probs = [self.emiss[st][act][obs] for obs in observation_set]
        next_observation = choices(observation_set, observation_probs, k=1)[0]
        return next_observation

    def reward_sampler(self, st):
        # sink always gives zero
        if st == self.sink_state:
            return 0
        if self.context == 0:
            if st == self.goal:
                return self.rewards
            elif st == self.detector:
                return choices([self.penalty, 0], [0.5, 0.5], k=1)[0]
            else:
                return 0
        elif self.context == 1:
            if st == self.goal:
                return self.rewards
            elif st == self.detector:
                return choices([self.penalty, 0], [0.2, 0.8], k=1)[0]
            else:
                return 0
        else:
            raise ValueError('Invalid context parameter.')

    def _reset_context_dependent_parts(self):
        """(Re)compute all quantities that depend on context."""
        # Initial state/dist
        self.initial_states = self.get_initial_state()
        self.initial_dist = self.get_initial_distribution()

        # Rewards / penalties / detector / goal
        self.rewards, self.goal = self.get_reward()
        self.penalty, self.detector = self.get_penalty()

        # Transitions
        # self.transition = self.get_transition()
        # self.check_trans()

        # Emission function
        # self.emiss = self.get_emission_function()
        # self.check_emission_function()

    def set_context(self, context):
        """Public setter to change context and recompute internals."""
        self.context = context
        self._reset_context_dependent_parts()
