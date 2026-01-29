import numpy as np
import cv2
from Gym_Example.archive_code.gym_env import VisionCtx2DNavEnv
# adjust import if needed


def run_episode_and_print_all(
    ctx=1,
    max_steps=200,
    video_path="cpomdp_demo.mp4",
    fps=10
):
    env = VisionCtx2DNavEnv(ctx=ctx)
    obs, _ = env.reset()

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    states = []
    actions = []
    observations = []

    # --------------------------------------------------
    # Video writer
    # --------------------------------------------------
    frame = env._render_obs()
    h, w, _ = frame.shape

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    done = False
    step = 0

    print("\n==============================")
    print(f"START EPISODE | CONTEXT = {ctx}")
    print("==============================\n")

    while not done and step < max_steps:
        # --------------------------------------------------
        # Simple deterministic + noise policy
        # --------------------------------------------------
        x, y, vx, vy, *_ = env.state
        goal = env.goal

        direction = goal - np.array([x, y])
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        action = 0.8 * direction + 0.2 * np.random.randn(2)
        action = np.clip(action, -1.0, 1.0)

        # --------------------------------------------------
        # Step
        # --------------------------------------------------
        next_obs, reward, done, _, _ = env.step(action)

        # --------------------------------------------------
        # Log
        # --------------------------------------------------
        states.append(env.state.copy())
        actions.append(action.copy())
        observations.append(next_obs.copy())

        # --------------------------------------------------
        # PRINT EVERYTHING (NO TRUNCATION)
        # --------------------------------------------------
        print(f"Step {step}")
        print("State:")
        print(env.state)
        print("Action:")
        print(action)
        print("Observation:")
        print(next_obs)
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print("-" * 50)

        # --------------------------------------------------
        # Render
        # --------------------------------------------------
        frame = env._render_obs()
        writer.write(frame)

        obs = next_obs
        step += 1

    writer.release()
    env.close()

    print("\n==============================")
    print(f"END EPISODE | steps = {step}")
    print(f"Video saved to: {video_path}")
    print("==============================\n")

    return (
        np.array(states),
        np.array(actions),
        np.array(observations)
    )


if __name__ == "__main__":
    # --------------------------------------------------
    # Run Context 1
    # --------------------------------------------------
    states_1, actions_1, obs_1 = run_episode_and_print_all(
        ctx=1,
        video_path="../ctx1_slippery_normal_npc.mp4"
    )

    # --------------------------------------------------
    # Run Context 2
    # --------------------------------------------------
    states_2, actions_2, obs_2 = run_episode_and_print_all(
        ctx=2,
        video_path="../ctx2_normal_aggressive_npc.mp4"
    )

    # --------------------------------------------------
    # FINAL FULL PRINT (ENTIRE TRAJECTORIES)
    # --------------------------------------------------
    print("\n========================================")
    print("FULL TRAJECTORY DUMP (CONTEXT 1)")
    print("========================================")
    print("States:")
    print(states_1)
    print("\nActions:")
    print(actions_1)
    print("\nObservations:")
    print(obs_1)

    print("\n========================================")
    print("FULL TRAJECTORY DUMP (CONTEXT 2)")
    print("========================================")
    print("States:")
    print(states_2)
    print("\nActions:")
    print(actions_2)
    print("\nObservations:")
    print(obs_2)
