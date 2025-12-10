from scipy.sparse import csr_matrix

class LazyObservableOperatorCP:
    """
    observable operator Ao|a^c
    cached by (context, obs, act)
    """

    def __init__(self, env):
        self.env = env
        self.cache = {}
        self.state_to_idx = {s:i for i,s in enumerate(env.states)}

    def get_operator(self, context, obs, act):
        key = (context, obs, act)
        if key in self.cache:
            return self.cache[key]

        rows, cols, data = [], [], []

        T = self.env.transitions[context]
        E = self.env.emissions[context]

        for st_prime in self.env.states:   # current state j
            if act not in T[st_prime]:
                continue
            if obs not in E[st_prime][act]:
                continue

            emiss_prob = E[st_prime][act][obs]
            if emiss_prob == 0:
                continue

            j = self.state_to_idx[st_prime]

            for st in T[st_prime][act]:   # next state i
                trans_prob = T[st_prime][act][st]
                if trans_prob == 0:
                    continue

                i = self.state_to_idx[st]
                value = emiss_prob * trans_prob
                rows.append(i)
                cols.append(j)
                data.append(value)

        if len(data) == 0:
            oo = csr_matrix((self.env.state_size, self.env.state_size))
        else:
            oo = csr_matrix((data, (rows, cols)),
                             shape=(self.env.state_size, self.env.state_size))

        self.cache[key] = oo
        return oo
