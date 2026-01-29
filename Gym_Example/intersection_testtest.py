import numpy as np
import cv2
from intersection_env import VisionCtxIntersectionEnv
# ↑ adjust filename if needed


def run_episode(
    ctx=1,
    max_steps=250,
    video_path="intersection_demo.mp4",
    fps=10
):
    env = VisionCtxIntersectionEnv(ctx=ctx)
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

    print("\n=======================================")
    print(f"START EPISODE | CONTEXT = {ctx}")
    print("=======================================\n")

    while not done and step < max_steps:
        # --------------------------------------------------
        # Simple road-following + goal-seeking policy
        # --------------------------------------------------
        x, y, vx, vy, *_ = env.state
        goal = env.goal

        # Longitudinal drive toward goal
        ax = 0.8 * np.sign(goal[0] - x)

        # Lateral stabilization toward road center
        ay = -0.5 * y

        action = np.clip(np.array([ax, ay]) + 0.2 * np.random.randn(2), -1.0, 1.0)

        # --------------------------------------------------
        # Step environment
        # --------------------------------------------------
        next_obs, reward, done, _, _ = env.step(action)

        # --------------------------------------------------
        # Log trajectories
        # --------------------------------------------------
        states.append(env.state.copy())
        actions.append(action.copy())
        observations.append(next_obs.copy())

        # --------------------------------------------------
        # PRINT EVERYTHING (no truncation)
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
        print("-" * 60)

        # --------------------------------------------------
        # Render + save frame
        # --------------------------------------------------
        frame = env._render_obs()
        writer.write(frame)

        step += 1

    writer.release()
    env.close()

    print("\n=======================================")
    print(f"END EPISODE | steps = {step}")
    print(f"Video saved to: {video_path}")
    print("=======================================\n")

    return (
        np.array(states),
        np.array(actions),
        np.array(observations)
    )


if __name__ == "__main__":
    # --------------------------------------------------
    # Run context 1: slippery road, normal driver
    # --------------------------------------------------
    states_1, actions_1, obs_1 = run_episode(
        ctx=1,
        video_path="ctx1_slippery_normal_intersection.mp4"
    )

    # --------------------------------------------------
    # Run context 2: normal road, aggressive driver
    # --------------------------------------------------
    states_2, actions_2, obs_2 = run_episode(
        ctx=2,
        video_path="ctx2_normal_aggressive_intersection.mp4"
    )

    # --------------------------------------------------
    # Final full trajectory dump (explicit)
    # --------------------------------------------------
    print("\n=======================================")
    print("FULL TRAJECTORY DUMP — CONTEXT 1")
    print("=======================================")
    print("States:\n", states_1)
    print("\nActions:\n", actions_1)
    print("\nObservations:\n", obs_1)

    print("\n=======================================")
    print("FULL TRAJECTORY DUMP — CONTEXT 2")
    print("=======================================")
    print("States:\n", states_2)
    print("\nActions:\n", actions_2)
    print("\nObservations:\n", obs_2)
