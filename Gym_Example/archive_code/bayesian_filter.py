def exact_logP_y_given_c_context_filter(self, context, z_seq, m_seq, act_seq):
    """
    Compute log P(y | c) using a context filter (no EKF).
    """

    log_prob = 0.0
    obs_history = []

    for t in range(self.T):

        a = act_seq[t]
        z = z_seq[t]
        m = m_seq[t]

        # ---------------------------
        # POLICY likelihood
        # ---------------------------
        if t == 0:
            inp = torch.zeros((1, 1, 2))
        else:
            inp = torch.tensor([obs_history], dtype=torch.float32)

        with torch.no_grad():
            probs, _ = self.policy(inp)

        a_idx = self.env.actions.index(a)
        pi = float(probs[0, a_idx])
        log_prob += math.log(pi + EPS)

        # ---------------------------
        # OBSERVATION likelihood
        # ---------------------------
        obs_logp = math.log(
            self.env.obs_likelihood(z, m, a, context) + EPS
        )
        log_prob += obs_logp

        obs_history.append([z, m])

    return log_prob


def logP_y_context_filter(self, z_seq, m_seq, act_seq):
    """
    Compute log P(y) = log Σ_c P(c) P(y|c)
    """
    log_terms = []

    for c in self.env.contexts:
        logPygc = self.exact_logP_y_given_c_context_filter(
            c, z_seq, m_seq, act_seq
        )
        log_terms.append(
            math.log(self.env.context_distribution[c] + EPS) + logPygc
        )

    # log-sum-exp for numerical stability
    max_log = max(log_terms)
    logP_y = max_log + math.log(
        sum(math.exp(l - max_log) for l in log_terms) + EPS
    )
    return logP_y

def posterior_and_entropy_context_filter(self, z_seq, m_seq, act_seq):
    """
    Returns:
        logP_c_given_y : dict {c: log P(c|y)}
        H             : entropy H(C|y)
    """

    log_joint = {}
    for c in self.env.contexts:
        logPygc = self.exact_logP_y_given_c_context_filter(
            c, z_seq, m_seq, act_seq
        )
        log_joint[c] = (
            math.log(self.env.context_distribution[c] + EPS)
            + logPygc
        )

    # normalize
    max_log = max(log_joint.values())
    Z = sum(math.exp(v - max_log) for v in log_joint.values()) + EPS
    logZ = max_log + math.log(Z)

    log_post = {c: log_joint[c] - logZ for c in log_joint}
    post = {c: math.exp(lp) for c, lp in log_post.items()}

    # entropy
    H = 0.0
    for p in post.values():
        H += -p * math.log(p + EPS)

    return log_post, H
