import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces


class VisionCtx2DNavEnv(gym.Env):
    """
    Vision-based CPOMDP with state-based observations.
    Agent always observes itself.
    NPC is observed only if inside camera range.
    """

    def __init__(self, ctx=1, img_size=64, horizon=200, camera_range=3.0):
        super().__init__()
        assert ctx in [1, 2]
        self.ctx = ctx
        self.img_size = img_size
        self.horizon = horizon
        self.camera_range = camera_range

        self.dt = 0.1

        # context-dependent dynamics
        if ctx == 1:
            self.sigma_dyn = 1.0      # slippery
            self.npc_aggressive = False
        else:
            self.sigma_dyn = 0.2      # normal road
            self.npc_aggressive = True

        # observation:
        # [ax, ay, avx, avy, npc_x, npc_y, npc_vx, npc_vy]
        high = np.full(8, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([
            -4.0, 0.0,    # agent x,y
            0.0, 0.0,     # agent vx,vy
            2.0, 0.0,     # npc x,y
            0.0, 0.0      # npc vx,vy
        ], dtype=np.float32)

        self.goal = np.array([5.0, 0.0])
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        ax, ay = np.clip(action, -1.0, 1.0)
        x, y, vx, vy, nx, ny, nvx, nvy = self.state

        # ---------- Agent dynamics ----------
        vx += self.dt * (ax + np.random.randn() * self.sigma_dyn)
        vy += self.dt * (ay + np.random.randn() * self.sigma_dyn)
        x += self.dt * vx
        y += self.dt * vy

        # ---------- NPC dynamics ----------
        if self.npc_aggressive:
            d = np.array([x - nx, y - ny])
            d /= (np.linalg.norm(d) + 1e-6)
            npc_acc = 0.8 * d + 0.3 * np.random.randn(2)
        else:
            npc_acc = 0.1 * np.random.randn(2)

        nvx += self.dt * npc_acc[0]
        nvy += self.dt * npc_acc[1]
        nx += self.dt * nvx
        ny += self.dt * nvy

        self.state = np.array([x, y, vx, vy, nx, ny, nvx, nvy])

        # ---------- Reward ----------
        reward = -1.0
        done = False

        if np.linalg.norm([x - nx, y - ny]) < 0.4:
            reward -= 100
            done = True

        if np.linalg.norm([x - self.goal[0], y - self.goal[1]]) < 0.3:
            reward += 100
            done = True

        self.step_count += 1
        if self.step_count >= self.horizon:
            done = True

        return self._get_obs(), reward, done, False, {}

    # =====================================================
    # Observation Model
    # =====================================================
    def _get_obs(self):
        """
        Agent always observes its own state.
        NPC is observed only if inside camera range.
        """
        x, y, vx, vy, nx, ny, nvx, nvy = self.state

        obs = np.full(8, np.nan, dtype=np.float32)

        # agent state always observed
        obs[0:4] = np.array([x, y, vx, vy])

        # NPC visibility check
        if np.linalg.norm([nx - x, ny - y]) <= self.camera_range:
            obs[4:] = np.array([nx, ny, nvx, nvy])

        return obs

    # =====================================================
    # Visualization (UNCHANGED)
    # =====================================================
    def _render_obs(self):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # ---------- Road texture ----------
        if self.ctx == 1:
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img[:] = 100 + noise
        else:
            img[:] = 120
            for i in range(0, self.img_size, 8):
                cv2.line(img, (i, 0), (i, self.img_size), (90, 90, 90), 1)
                cv2.line(img, (0, i), (self.img_size, i), (90, 90, 90), 1)

        # ---------- Coordinate transform ----------
        scale = 5.0  # pixels per world unit
        cx = self.img_size // 2
        cy = self.img_size // 2

        x, y, _, _, nx, ny, _, _ = self.state

        # Agent is always at the center
        ax_px, ay_px = cx, cy

        # NPC relative position
        dx = nx - x
        dy = ny - y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        nx_px = int(cx + dx * scale)
        ny_px = int(cy + dy * scale)

        # ---------- Draw circular camera range ----------
        cam_radius_px = int(self.camera_range * scale)
        cv2.circle(
            img,
            (cx, cy),
            cam_radius_px,
            (180, 180, 180),
            1  # thin outline
        )

        # ---------- Draw agent ----------
        cv2.circle(img, (ax_px, ay_px), 4, (0, 0, 255), -1)

        # ---------- Draw NPC ONLY if inside circular range ----------
        if dist <= self.camera_range:
            if 0 <= nx_px < self.img_size and 0 <= ny_px < self.img_size:
                cv2.circle(img, (nx_px, ny_px), 4, (255, 0, 0), -1)

        return img

    def render(self):
        cv2.imshow("Vision CPOMDP", self._render_obs())
        cv2.waitKey(1)
