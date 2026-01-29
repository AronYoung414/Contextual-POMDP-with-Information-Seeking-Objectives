import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt


# ============================================================
# 1) Generative CPOMDP model (POMDP with hidden context)
#    Hidden state = (x, c). Observation = noisy scalar z.
# ============================================================

@dataclass(frozen=True)
class Action:
    name: str


A_LEFT = Action("L")
A_RIGHT = Action("R")
A_OBSERVE = Action("O")
ACTIONS = [A_LEFT, A_RIGHT, A_OBSERVE]


class LightDarkCPOMDP:
    """
    Continuous 1D Light-Dark CPOMDP.
      - Hidden state: (x, c), where c in {0,1} is context (fixed in episode).
      - Actions: left/right/observe.
      - Transition:
          L/R: x <- x +/- step + N(0, dyn_var)
          O  : x <- x + N(0, dyn_var_obs)   (often small; you can set to 0)
      - Observation:
          z ~ N(x, sigma^2(x,c))  where sigma^2 depends on light/dark region under context c
      - Reward:
          +R_goal if x in reward interval (context-dependent)
          -R_pen  if x in penalty interval (context-dependent)
          else 0
    """

    def __init__(
            self,
            T: int = 30,
            x_min: float = -6.0,
            x_max: float = 6.0,
            step: float = 0.5,
            dyn_var_move: float = 0.05,
            dyn_var_obs: float = 0.01,
            light_var: float = 1.0,  # matches your plot vars
            dark_var: float = 8.0,
            reward_regions: Optional[Dict[int, Dict[str, Tuple[float, float]]]] = None,
            R_goal: float = 1.0,
            R_pen: float = 1.0,
            p_context: Tuple[float, float] = (0.5, 0.5),
            seed: int = 0,
    ):
        self.T = T
        self.x_min = x_min
        self.x_max = x_max
        self.step = step
        self.dyn_var_move = dyn_var_move
        self.dyn_var_obs = dyn_var_obs
        self.light_var = light_var
        self.dark_var = dark_var
        self.R_goal = R_goal
        self.R_pen = R_pen
        self.p_context = np.array(p_context, dtype=float)
        self.p_context /= self.p_context.sum()
        self.rng = np.random.default_rng(seed)

        if reward_regions is None:
            reward_regions = {
                0: {"reward": (1, 5), "penalty": (-5, 0)},
                1: {"reward": (-5, -1), "penalty": (0, 5)},
            }
        self.reward_regions = reward_regions

    # --- Light/Dark rule depends on context ---
    def is_light(self, x: float, c: int) -> bool:
        if c == 0:
            return x > 0
        else:
            return x < 0

    def obs_var(self, x: float, c: int) -> float:
        return self.light_var if self.is_light(x, c) else self.dark_var

    # --- Reward based on x interval (context-dependent) ---
    def reward(self, x: float, c: int) -> float:
        r_int = self.reward_regions[c]["reward"]
        p_int = self.reward_regions[c]["penalty"]
        if r_int[0] <= x <= r_int[1]:
            return +self.R_goal
        if p_int[0] <= x <= p_int[1]:
            return -self.R_pen
        return 0.0

    def clamp(self, x: float) -> float:
        return float(np.clip(x, self.x_min, self.x_max))

    # --- Generative model: sample next hidden state and observation given action ---
    def step_generative(self, state: Tuple[float, int], action: Action) -> Tuple[Tuple[float, int], float, float]:
        x, c = state

        if action.name == "L":
            x2 = x - self.step + self.rng.normal(0.0, math.sqrt(self.dyn_var_move))
        elif action.name == "R":
            x2 = x + self.step + self.rng.normal(0.0, math.sqrt(self.dyn_var_move))
        else:  # Observe
            x2 = x + self.rng.normal(0.0, math.sqrt(self.dyn_var_obs))

        x2 = self.clamp(x2)

        var = self.obs_var(x2, c)
        z = x2 + self.rng.normal(0.0, math.sqrt(var))
        r = self.reward(x2, c)
        return (x2, c), z, r

    def sample_initial_state(self, x0_mean=0.0, x0_std=2.0) -> Tuple[float, int]:
        c = int(self.rng.choice([0, 1], p=self.p_context))
        x = self.clamp(self.rng.normal(x0_mean, x0_std))
        return (x, c)


# ============================================================
# 2) Particle belief update (bootstrap filter with likelihood weights)
# ============================================================

def normal_pdf(x, mu, var):
    # Stable-ish scalar normal pdf
    return (1.0 / math.sqrt(2.0 * math.pi * var)) * math.exp(-0.5 * (x - mu) ** 2 / var)


def belief_update_particles(
        model: LightDarkCPOMDP,
        particles: np.ndarray,  # shape (N,2): [x, c]
        action: Action,
        obs: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Propagate each particle through transition, then weight by observation likelihood, then resample.
    """
    N = particles.shape[0]
    new_particles = np.zeros_like(particles)

    # propagate and compute weights
    ws = np.zeros(N, dtype=float)
    for i in range(N):
        x, c = float(particles[i, 0]), int(particles[i, 1])
        (x2, c2), z_pred, _r = model.step_generative((x, c), action)
        # likelihood p(obs | x2,c2)
        var = model.obs_var(x2, c2)
        w = normal_pdf(obs, x2, var)  # z ~ N(x2, var)
        new_particles[i, 0] = x2
        new_particles[i, 1] = c2
        ws[i] = w

    s = ws.sum()
    if not np.isfinite(s) or s <= 0:
        # fall back to uniform if numerical issues
        idx = rng.integers(0, N, size=N)
        return new_particles[idx]

    ws /= s
    idx = rng.choice(np.arange(N), size=N, replace=True, p=ws)
    return new_particles[idx]


# ============================================================
# 3) POMCP (UCT in belief tree)
# ============================================================

class TreeNode:
    __slots__ = ("N", "Q", "children")  # keep it light

    def __init__(self):
        self.N = 0
        self.Q = 0.0
        self.children: Dict[str, "TreeNode"] = {}


def obs_key(z: float, bin_width: float = 0.25) -> str:
    # Discretize continuous observations into bins for tree branching
    b = int(math.floor(z / bin_width))
    return f"zbin:{b}"


class POMCPSolver:
    def __init__(
            self,
            model: LightDarkCPOMDP,
            n_sims: int = 2000,
            c_uct: float = 1.5,
            gamma: float = 1.0,
            rollout_depth: Optional[int] = None,
            obs_bin_width: float = 0.25,
            seed: int = 1,
    ):
        self.model = model
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.gamma = gamma
        self.rollout_depth = rollout_depth if rollout_depth is not None else model.T
        self.obs_bin_width = obs_bin_width
        self.rng = np.random.default_rng(seed)

    def uct_select(self, node: TreeNode) -> Action:
        # choose action maximizing UCB
        best_a = None
        best_val = -1e18
        for a in ACTIONS:
            key = f"a:{a.name}"
            child = node.children.get(key)
            if child is None or child.N == 0:
                return a  # expand untried action
            ucb = child.Q + self.c_uct * math.sqrt(math.log(node.N + 1) / (child.N))
            if ucb > best_val:
                best_val = ucb
                best_a = a
        return best_a if best_a is not None else self.rng.choice(ACTIONS)

    def rollout_policy(self, _state: Tuple[float, int]) -> Action:
        # simple default rollout
        return self.rng.choice(ACTIONS)

    def simulate(self, node: TreeNode, state: Tuple[float, int], depth: int) -> float:
        if depth <= 0:
            return 0.0

        if node.N == 0:
            # leaf: do rollout from here
            return self.rollout(state, depth)

        a = self.uct_select(node)
        a_key = f"a:{a.name}"
        if a_key not in node.children:
            node.children[a_key] = TreeNode()
        a_node = node.children[a_key]

        (s2, _c2), z, r = self.model.step_generative(state, a)
        o_key = obs_key(z, self.obs_bin_width)
        if o_key not in a_node.children:
            a_node.children[o_key] = TreeNode()
        o_node = a_node.children[o_key]

        ret = r + self.gamma * self.simulate(o_node, s2, depth - 1)

        # backprop updates
        node.N += 1
        a_node.N += 1
        # incremental mean update for Q
        a_node.Q += (ret - a_node.Q) / a_node.N
        return ret

    def rollout(self, state: Tuple[float, int], depth: int) -> float:
        total = 0.0
        disc = 1.0
        s = state
        for d in range(depth):
            a = self.rollout_policy(s)
            s, z, r = self.model.step_generative(s, a)
            total += disc * r
            disc *= self.gamma
        return total

    def plan(self, belief_particles: np.ndarray, depth: int) -> Action:
        root = TreeNode()
        # run simulations
        for _ in range(self.n_sims):
            i = self.rng.integers(0, belief_particles.shape[0])
            x = float(belief_particles[i, 0])
            c = int(belief_particles[i, 1])
            self.simulate(root, (x, c), depth)

        # choose best action by highest Q among tried actions
        best_a = None
        best_q = -1e18
        for a in ACTIONS:
            child = root.children.get(f"a:{a.name}")
            if child is None or child.N == 0:
                continue
            if child.Q > best_q:
                best_q = child.Q
                best_a = a
        return best_a if best_a is not None else self.rng.choice(ACTIONS)


# ============================================================
# 4) Run one episode: belief tracking + POMCP planning
# ============================================================

def run_episode(
        model: LightDarkCPOMDP,
        solver: POMCPSolver,
        N_particles: int = 1000,
        x0_mean: float = 0.0,
        x0_std: float = 2.0,
        seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # Sample true hidden initial state
    true_state = model.sample_initial_state(x0_mean=x0_mean, x0_std=x0_std)

    # Initialize belief as particles over (x,c)
    particles = np.zeros((N_particles, 2), dtype=float)
    for i in range(N_particles):
        # prior over context + x
        s = model.sample_initial_state(x0_mean=x0_mean, x0_std=x0_std)
        particles[i, 0] = s[0]
        particles[i, 1] = s[1]

    total_return = 0.0
    traj = []

    for t in range(model.T):
        a = solver.plan(particles, depth=(model.T - t))
        true_state, z, r = model.step_generative(true_state, a)

        # belief update using observation z
        particles = belief_update_particles(model, particles, a, z, rng)

        total_return += r
        traj.append((t, a.name, float(true_state[0]), int(true_state[1]), float(z), float(r)))

    return total_return, traj


def plot_pomcp_training_curve(
        model,
        solver_class,
        sim_budgets,
        n_episodes=10,
        n_particles=1000,
        seed=0,
):
    """
    Plot expected return vs number of POMCP simulations.

    Args:
        model: LightDarkCPOMDP instance
        solver_class: POMCPSolver class
        sim_budgets: list of ints (number of simulations)
        n_episodes: number of episodes to average over
        n_particles: number of belief particles
    """
    rng = np.random.default_rng(seed)

    mean_returns = []
    std_returns = []

    for sims in sim_budgets:
        episode_returns = []

        for ep in range(n_episodes):
            solver = solver_class(
                model,
                n_sims=sims,
                seed=seed + ep,
            )

            total_return, _ = run_episode(
                model,
                solver,
                N_particles=n_particles,
                seed=seed + 1000 * ep,
            )
            episode_returns.append(total_return)

        episode_returns = np.array(episode_returns)
        mean_returns.append(episode_returns.mean())
        std_returns.append(episode_returns.std())

        print(
            f"Sims={sims:4d} | "
            f"Mean return={episode_returns.mean():.3f} ± {episode_returns.std():.3f}"
        )

    # ==========================
    # Plot
    # ==========================
    plt.figure(figsize=(7, 4))

    plt.plot(sim_budgets, mean_returns, marker="o", linewidth=2)
    plt.fill_between(
        sim_budgets,
        np.array(mean_returns) - np.array(std_returns),
        np.array(mean_returns) + np.array(std_returns),
        alpha=0.25,
    )

    plt.xscale("log")
    plt.xlabel("Number of POMCP simulations")
    plt.ylabel("Expected total return")
    plt.title("POMCP planning improves with simulation budget")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Match your plotting parameters
    model = LightDarkCPOMDP(
        T=30,
        light_var=1.0,
        dark_var=8.0,
        reward_regions={
            0: {"reward": (1, 5), "penalty": (-5, 0)},
            1: {"reward": (-5, -1), "penalty": (0, 5)},
        },
        R_goal=1.0,
        R_pen=1.0,
        step=0.5,
        dyn_var_move=0.05,
        dyn_var_obs=0.01,
        p_context=(0.5, 0.5),
        seed=0,
    )

    solver = POMCPSolver(
        model,
        n_sims=2500,  # increase for better returns
        c_uct=1.5,
        gamma=1.0,
        obs_bin_width=0.25,
        seed=1,
    )

    ret, traj = run_episode(model, solver, N_particles=1200, seed=7)
    print("Total return:", ret)
    print("First 10 steps:")
    for row in traj[:10]:
        print(row)

    sim_budgets = [100, 300, 700, 1500, 3000]

    plot_pomcp_training_curve(
        model=model,
        solver_class=POMCPSolver,
        sim_budgets=sim_budgets,
        n_episodes=12,
        n_particles=1200,
        seed=0,
    )


class POMCPPolicy:
    def __init__(
            self,
            model: LightDarkCPOMDP,
            solver: POMCPSolver,
            n_particles=1000,
            x0_mean=0.0,
            x0_std=2.0,
            seed=0,
    ):
        self.model = model
        self.solver = solver
        self.n_particles = n_particles
        self.x0_mean = x0_mean
        self.x0_std = x0_std
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        # initialize belief particles
        self.particles = np.zeros((self.n_particles, 2))
        for i in range(self.n_particles):
            x, c = self.model.sample_initial_state(
                self.x0_mean, self.x0_std
            )
            self.particles[i] = [x, c]

        self.last_action = None
        self.t = 0

    def act(self, obs):
        """
        obs: scalar observation z_t (None at t=0)
        """
        if obs is not None:
            self.particles = belief_update_particles(
                self.model,
                self.particles,
                self.last_action,
                obs,
                self.rng,
            )

        action = self.solver.plan(
            self.particles,
            depth=self.model.T - self.t,
        )

        self.last_action = action
        self.t += 1
        return action.name

    def update_belief(self, obs):
        self.particles = belief_update_particles(
            self.model,
            self.particles,
            self.last_action,
            obs,
            self.rng,
        )

    def plan_action(self):
        action = self.solver.plan(
            self.particles,
            depth=self.model.T - self.t,
        )
        self.t += 1
        return action


def belief_signature(particles):
    """
    Compact belief summary for caching
    """
    x_mean = particles[:, 0].mean()
    x_std = particles[:, 0].std()
    p_c1 = (particles[:, 1] == 1).mean()
    return (
        round(x_mean, 2),
        round(x_std, 2),
        round(p_c1, 2),
    )


class CachedPOMCPPolicy:
    def __init__(self, pomcp_policy):
        self.pomcp = pomcp_policy
        self.cache = {}

    def reset(self):
        self.pomcp.reset()

    def act(self, obs):
        # 1) Update belief ONLY if we have a previous action
        if obs is not None and self.pomcp.last_action is not None:
            self.pomcp.update_belief(obs)

        # 2) Compute belief key
        key = belief_signature(self.pomcp.particles)

        # 3) Plan ONLY if unseen belief
        if key not in self.cache:
            action = self.pomcp.plan_action()
            self.cache[key] = action

        # 4) Record last action
        self.pomcp.last_action = self.cache[key]

        return self.cache[key].name.lower()

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.cache, f)
