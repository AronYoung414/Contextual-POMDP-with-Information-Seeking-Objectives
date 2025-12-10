import numpy as np
from random import choices, seed
import random

class EnvironmentCPOMDP:
    """
    Clean CPOMDP Environment:
      - No mutation
      - All transitions/emissions/rewards stored per context
      - All sampling functions take (context) as explicit parameter
    """

    def __init__(self, stoPar=0.1, obsNoise=0.1):
        self.contexts = [0, 1]
        self.context_distribution = {0: 0.5, 1: 0.5}

        self.stoPar = stoPar
        self.obsNoise = obsNoise

        self.states = [0, 1, 2, 3, 4, 5, 6, 7]  # include sink=7
        self.sink_state = 7
        self.state_size = len(self.states)

        self.actions = ['r', 'l', 's']
        self.action_size = len(self.actions)

        self.observations = ['0','L','R','H','PAD']
        self.start_idx = 4  # PAD token for empty history

        # build CPOMDP components
        self.initial_states = self._build_initial_states()
        self.initial_dist = self._build_initial_dist()

        self.reward_func = self._build_rewards()
        self.detectors = self._build_detectors()

        self.transitions = self._build_transitions()
        self.emissions = self._build_emissions()

    def seed(self, sd):
        random.seed(sd)
        np.random.seed(sd)

    # -------------------------------------------------------
    # INITIAL STATE PER CONTEXT
    # -------------------------------------------------------
    def _build_initial_states(self):
        """
        context 0 → initial state = 4
        context 1 → initial state = 2
        """
        return {
            0: [4],
            1: [2]
        }

    def _build_initial_dist(self):
        init = {}
        for c in self.contexts:
            mu = np.zeros((self.state_size, 1))
            for s in self.initial_states[c]:
                mu[self.states.index(s), 0] = 1.0
            init[c] = mu
        return init

    # -------------------------------------------------------
    # REWARD STRUCTURE PER CONTEXT
    # -------------------------------------------------------
    def _build_rewards(self):
        """
        context 0:
          goal=6 reward=3
          detector=5 penalty=-2
        context 1:
          goal=0 reward=5
          detector=1 penalty=-10
        """
        return {
            0: {"goal": 6, "reward": 3, "detector": 1, "penalty": -2},
            1: {"goal": 0, "reward": 5, "detector": 5, "penalty": -10}
        }

    def _build_detectors(self):
        return { c: self.reward_func[c]["detector"] for c in self.contexts }

    # -------------------------------------------------------
    # TRANSITIONS PER CONTEXT
    # -------------------------------------------------------
    def _build_transitions(self):
        transitions = {}
        for c in self.contexts:
            T = {}
            goal = self.reward_func[c]["goal"]
            for st in self.states:
                T[st] = {}
                for act in self.actions:

                    # sink → sink
                    if st == self.sink_state:
                        T[st][act] = { self.sink_state: 1.0 }
                        continue

                    # goal → sink
                    if st == goal:
                        T[st][act] = { self.sink_state: 1.0 }
                        continue

                    # normal transitions
                    T[st][act] = {}

                    if act == 's':
                        T[st][act][st] = 1.0

                    else:
                        if act == 'l': next_st = st - 1
                        elif act == 'r': next_st = st + 1

                        if next_st in self.states:
                            T[st][act][next_st] = 1 - self.stoPar
                            T[st][act][st] = self.stoPar
                        else:
                            T[st][act][st] = 1.0
            transitions[c] = T
        return transitions

    # -------------------------------------------------------
    # EMISSIONS PER CONTEXT
    # -------------------------------------------------------
    def _build_emissions(self):
        emiss = {}
        for c in self.contexts:
            E = {}
            det = self.reward_func[c]["detector"]
            for st in self.states:
                E[st] = {}
                for act in self.actions:

                    if st == self.sink_state:
                        E[st][act] = {'0': 1.0}
                        continue

                    if act != 's':
                        E[st][act] = {'0': 1.0}
                        continue

                    # sensing
                    E[st][act] = {}
                    if st > det:
                        E[st][act]['L'] = 1 - self.obsNoise
                        E[st][act]['0'] = self.obsNoise
                    elif st == det:
                        E[st][act]['H'] = 1 - self.obsNoise
                        E[st][act]['0'] = self.obsNoise
                    else:
                        E[st][act]['R'] = 1 - self.obsNoise
                        E[st][act]['0'] = self.obsNoise
            emiss[c] = E
        return emiss

    # -------------------------------------------------------
    # SAMPLERS
    # -------------------------------------------------------
    def next_state_sampler(self, st, act, context):
        d = self.transitions[context][st][act]
        supp = list(d.keys())
        probs = [d[s] for s in supp]
        return choices(supp, probs, k=1)[0]

    def observation_function_sampler(self, st, act, context):
        d = self.emissions[context][st][act]
        supp = list(d.keys())
        probs = [d[o] for o in supp]
        return choices(supp, probs, k=1)[0]

    def reward_sampler(self, st, context):
        if st == self.sink_state:
            return 0
        goal = self.reward_func[context]["goal"]
        det = self.reward_func[context]["detector"]
        R = self.reward_func[context]["reward"]
        P = self.reward_func[context]["penalty"]

        if st == goal:
            return R
        if st == det:
            # stochastic penalty
            return choices([P, 0], [0.5, 0.5], k=1)[0]
        return 0
