import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces


class VisionCtxIntersectionEnv(gym.Env):
    """
    CPOMDP: Two drivers with an intersection / merging lane.
    Hard rectangular boundary, road geometry, partial observations.
    """

    def __init__(self, ctx=1, img_size=64, horizon=250, camera_range=10.0):
        super().__init__()
        assert ctx in [1, 2]
        self.ctx = ctx
        self.img_size = img_size
        self.horizon = horizon
        self.camera_range = camera_range
        self.dt = 0.1

        # -------------------------------
        # World boundary (HARD)
        # -------------------------------
        self.x_lim = (-10.0, 10.0)
        self.y_lim = (-10.0, 10.0)

        # -------------------------------
        # Road geometry
        # -------------------------------
        self.road_half_width = 1.5
        self.intersection_half = 1.5

        # -------------------------------
        # Context-dependent dynamics
        # -------------------------------
        if ctx == 1:
            self.sigma_dyn = 0.25      # slippery road
            self.npc_aggressive = False
        else:
            self.sigma_dyn = 0.05     # normal road
            self.npc_aggressive = True

        # Observation: agent always known, NPC gated by camera
        high = np.full(8, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))

        self.reset()

    # ======================================================
    # Reset
    # ======================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Agent on main road (left → right)
        # NPC on side road (bottom → up)
        self.state = np.array([
            -8.0, 0.0,     # agent x,y
            1.0, 0.0,      # agent vx,vy
            0.0, -8.0,     # npc x,y
            0.0, 1.0       # npc vx,vy
        ], dtype=np.float32)

        self.goal = np.array([8.0, 0.0])
        self.step_count = 0

        return self._get_obs(), {}

    # ======================================================
    # Step
    # ======================================================
    def step(self, action):
        ax, ay = np.clip(action, -1.0, 1.0)
        x, y, vx, vy, nx, ny, nvx, nvy = self.state

        # -------- Agent dynamics --------
        # velocity update (deterministic)
        vx += self.dt * ax
        vy += self.dt * ay

        # position update (stochastic)
        x += self.dt * vx + np.random.randn() * self.sigma_dyn
        y += self.dt * vy + np.random.randn() * self.sigma_dyn

        # -------- NPC dynamics --------
        if self.npc_aggressive:
            npc_acc = self._npc_aggressive_potential_acc(
                x=nx, y=ny, vx=nvx, vy=nvy,
                ax=x, ay=y
            )
        else:
            # Normal NPC
            npc_acc = self._npc_normal_potential_acc(
                x=nx, y=ny, vx=nvx, vy=nvy,
                ax=x, ay=y
            )

        # -------- NPC dynamics (NEW) --------
        nvx += self.dt * npc_acc[0]
        nvy += self.dt * npc_acc[1]

        nx += self.dt * nvx + np.random.randn() * self.sigma_dyn
        ny += self.dt * nvy + np.random.randn() * self.sigma_dyn

        # -------- Hard boundary reflection --------
        x, vx = self._reflect(x, vx, self.x_lim)
        y, vy = self._reflect(y, vy, self.y_lim)
        nx, nvx = self._reflect(nx, nvx, self.x_lim)
        ny, nvy = self._reflect(ny, nvy, self.y_lim)

        self.state = np.array([x, y, vx, vy, nx, ny, nvx, nvy])

        # -------- Reward --------
        reward = -1.0

        # Off-road penalty
        if not self._on_road(x, y):
            reward -= 2.0

        # Collision
        if np.linalg.norm([x - nx, y - ny]) < 2:
            reward -= 100
            done = True
        # Goal
        elif np.linalg.norm([x - self.goal[0], y - self.goal[1]]) < 1:
            reward += 100
            done = True
        else:
            done = False

        self.step_count += 1
        if self.step_count >= self.horizon:
            done = True

        return self._get_obs(), reward, done, False, {}

    # ======================================================
    # Normal NPC model
    # ======================================================
    def _npc_normal_potential_acc(self, x, y, vx, vy, ax, ay):
        """
        Potential-field normal NPC:
        - move upward slowly
        - stay near x=0 (vertical lane)
        - avoid collision with agent
        """
        # Agent state
        # (ax, ay are agent position; names chosen to avoid collision with accel)
        p_a = np.array([ax, ay], dtype=np.float32)

        # NPC state
        p_n = np.array([x, y], dtype=np.float32)
        v_n = np.array([vx, vy], dtype=np.float32)

        # ---------- potentials / gains ----------
        g_n = np.array([0.0, 8.0], dtype=np.float32)  # NPC "destination" at top

        k_g = 0.08  # goal attraction strength (smaller => slower progress)
        k_lane = 0.30  # keep x near 0 (vertical road center)
        k_rep = 0.80  # repulsion strength
        d0 = 2.0  # repulsion range (world units)

        k_d = 0.30  # damping
        k_s = 0.40  # speed tracking gain
        v_des = np.array([0.0, 0.8], dtype=np.float32)  # slow upward speed

        # ---------- gradient terms ----------
        # Goal attraction force: -∇U_goal
        f_goal = -k_g * (p_n - g_n)

        # Lane keeping force: -∇U_lane = [-k_lane*x, 0]
        f_lane = np.array([-k_lane * p_n[0], 0.0], dtype=np.float32)

        # Repulsion force: -∇U_rep (only when close)
        diff = p_n - p_a
        dist = float(np.linalg.norm(diff) + 1e-6)

        f_rep = np.zeros(2, dtype=np.float32)
        if dist < d0:
            # derivative of 0.5*k_rep*(1/d - 1/d0)^2 w.r.t position
            # f_rep points away from agent
            coeff = k_rep * (1.0 / dist - 1.0 / d0) * (1.0 / (dist ** 3))
            f_rep = coeff * diff

        # Damping and speed tracking
        f_damp = -k_d * v_n
        f_speed = k_s * (v_des - v_n)

        a_npc = f_goal + f_lane + f_rep + f_damp + f_speed

        # Clip acceleration to keep it realistic
        a_npc = np.clip(a_npc, -1.0, 1.0)
        return a_npc

    # ======================================================
    # Aggressive NPC model
    # ======================================================

    def _npc_aggressive_potential_acc(self, x, y, vx, vy, ax, ay):
        """
        Potential-field aggressive NPC:
        - actively seeks collision with the agent
        - weakly respects the road
        """
        # Positions and velocity
        p_n = np.array([x, y], dtype=np.float32)
        p_a = np.array([ax, ay], dtype=np.float32)
        v_n = np.array([vx, vy], dtype=np.float32)

        # ---------- gains ----------
        k_hit = 0.9  # attraction to agent (main aggressiveness)
        k_lane = 0.08  # weak lane constraint
        k_d = 0.25  # damping
        k_s = 0.30  # speed tracking

        # Desired velocity points toward agent
        diff = p_a - p_n
        dist = float(np.linalg.norm(diff) + 1e-6)
        dir_to_agent = diff / dist

        v_des = 1.5 * dir_to_agent  # aggressive speed

        # ---------- forces ----------
        # Hit-the-agent force
        f_hit = k_hit * diff

        # Weak lane keeping (stay roughly on vertical road)
        f_lane = np.array([-k_lane * p_n[0], 0.0], dtype=np.float32)

        # Damping
        f_damp = -k_d * v_n

        # Speed tracking (helps interception)
        f_speed = k_s * (v_des - v_n)

        a_npc = f_hit + f_lane + f_damp + f_speed

        # Clip acceleration
        a_npc = np.clip(a_npc, -1.5, 1.5)
        return a_npc

    # ======================================================
    # Observation model
    # ======================================================
    def _get_obs(self):
        x, y, vx, vy, nx, ny, nvx, nvy = self.state
        obs = np.full(8, np.nan, dtype=np.float32)
        obs[:4] = [x, y, vx, vy]

        if np.linalg.norm([nx - x, ny - y]) <= self.camera_range:
            obs[4:] = [nx, ny, nvx, nvy]

        return obs

    # ======================================================
    # Road check
    # ======================================================
    def _on_road(self, x, y):
        on_main = abs(y) <= self.road_half_width
        on_side = abs(x) <= self.road_half_width
        return on_main or on_side

    # ======================================================
    # Hard reflection utility
    # ======================================================
    def _reflect(self, pos, vel, bounds):
        if pos < bounds[0]:
            pos = bounds[0]
            vel = -vel
        elif pos > bounds[1]:
            pos = bounds[1]
            vel = -vel
        return pos, vel

    # ======================================================
    # Rendering (demo only)
    # ======================================================
    def _render_obs(self):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # -----------------------------
        # World → pixel transform
        # -----------------------------
        x_min, x_max = self.x_lim
        y_min, y_max = self.y_lim

        scale_x = self.img_size / (x_max - x_min)
        scale_y = self.img_size / (y_max - y_min)

        def world_to_px(x, y):
            px = int((x - x_min) * scale_x)
            py = int(self.img_size - (y - y_min) * scale_y)
            return px, py

        # -----------------------------
        # Background (grass)
        # -----------------------------
        img[:] = (25, 25, 25)

        # -----------------------------
        # Draw roads (fixed world)
        # -----------------------------
        road_px = int(self.road_half_width * scale_y)
        inter_px = int(self.intersection_half * scale_y)

        # Intersection box
        x0, y0 = world_to_px(0, 0)
        cv2.rectangle(
            img,
            (x0 - inter_px, y0 - inter_px),
            (x0 + inter_px, y0 + inter_px),
            (95, 95, 95),
            -1
        )

        # Main horizontal road
        cv2.rectangle(
            img,
            (0, y0 - road_px),
            (self.img_size, y0 + road_px),
            (80, 80, 80),
            -1
        )

        # Side vertical road
        cv2.rectangle(
            img,
            (x0 - road_px, 0),
            (x0 + road_px, self.img_size),
            (80, 80, 80),
            -1
        )

        # -----------------------------
        # Lane markings
        # -----------------------------
        dash_len = 6
        gap = 6

        # Horizontal dashed center line
        for px in range(0, self.img_size, dash_len + gap):
            cv2.line(
                img,
                (px, y0),
                (px + dash_len, y0),
                (180, 180, 180),
                1
            )

        # Vertical dashed center line
        for py in range(0, self.img_size, dash_len + gap):
            cv2.line(
                img,
                (x0, py),
                (x0, py + dash_len),
                (180, 180, 180),
                1
            )

        # -----------------------------
        # Draw camera range
        # -----------------------------
        x, y, _, _, nx, ny, _, _ = self.state
        ax_px, ay_px = world_to_px(x, y)
        cam_r_px = int(self.camera_range * scale_x)

        cv2.circle(
            img,
            (ax_px, ay_px),
            cam_r_px,
            (160, 160, 160),
            1
        )

        # -----------------------------
        # Draw agent (smaller)
        # -----------------------------
        cv2.circle(
            img,
            (ax_px, ay_px),
            3,  # smaller radius
            (0, 80, 255),  # blue
            -1
        )

        # -----------------------------
        # Draw NPC (smaller, only if visible)
        # -----------------------------
        if np.linalg.norm([nx - x, ny - y]) <= self.camera_range:
            nx_px, ny_px = world_to_px(nx, ny)
            cv2.circle(
                img,
                (nx_px, ny_px),
                3,  # smaller radius
                (255, 80, 80),  # red
                -1
            )

        return img

    def render(self):
        cv2.imshow("Intersection CPOMDP", self._render_obs())
        cv2.waitKey(1)
