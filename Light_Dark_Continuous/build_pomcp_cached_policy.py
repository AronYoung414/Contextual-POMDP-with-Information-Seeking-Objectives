from POMCP_baseline import (
    LightDarkCPOMDP,
    POMCPSolver,
    POMCPPolicy,
    CachedPOMCPPolicy,
)

from light_dark_environment import ContinuousLightDarkPOMDP

import numpy as np
import pickle

# ===== Environment =====
env = ContinuousLightDarkPOMDP()

# ===== POMCP =====
model = LightDarkCPOMDP()
solver = POMCPSolver(
    model,
    n_sims=3000,        # slow but only once
)

pomcp = POMCPPolicy(
    model=model,
    solver=solver,
    n_particles=1500,
)

cached_policy = CachedPOMCPPolicy(pomcp)

# ===== Offline rollouts =====
NUM_EPISODES = 10000
HORIZON = 20

episode_returns = []

for ep in range(NUM_EPISODES):
    cached_policy.reset()

    context = np.random.choice(
        env.contexts,
        p=[env.context_distribution[c] for c in env.contexts]
    )

    x = env.sample_initial_state(context)
    obs = None
    total_reward = 0.0

    for t in range(HORIZON):
        a = cached_policy.act(obs)
        x, z, r = env.step(x, a, context)

        total_reward += r
        obs = z

    episode_returns.append(total_reward)
    print(f"Episode {ep+1}/{NUM_EPISODES} | Return = {total_reward:.3f}")


# ===== Save =====
# cached_policy.save("data_pomcp/pomcp_cached_policy.pkl")
# print("Saved pomcp_cached_policy.pkl")

with open("data_pomcp/pomcp_cached_policy.pkl", "wb") as f:
    pickle.dump(
        {
            "policy": cached_policy.cache,
            "returns": episode_returns,
        },
        f
    )

print("Saved cached policy and returns.")

