import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from src.envs.quadruped_env import QuadrupedEnv
from src.evaluation.compare_dr import (
    load_observation_stats,
    normalize_observation,
)

MODEL = Path("models/walk_dr.zip")
OUTPUT = Path("results/turning_demo.mp4")
FPS = 20
SECONDS = 20

# (start, end, forward speed, yaw rate, label)
SEGMENTS = [
    (0, 5, 0.60, 0.0, "STRAIGHT"),
    (5, 10, 0.60, 0.30, "LEFT TURN"),
    (10, 15, 0.60, -0.30, "RIGHT TURN"),
    (15, 20, 0.60, 0.0, "STRAIGHT"),
]

def command_at(t):
    for start, end, vx, yaw, label in SEGMENTS:
        if start <= t < end:
            return np.array([vx, 0.0, yaw]), label
    return np.array([0.60, 0.0, 0.0]), "STRAIGHT"

def main():
    import imageio.v2 as imageio

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    mean, var = load_observation_stats(MODEL)
    policy = PPO.load(str(MODEL), device="cpu")
    env = QuadrupedEnv(dr_config=None)

    obs, _ = env.reset(seed=42)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.8
    camera.azimuth = 135
    camera.elevation = -25

    renderer = mujoco.Renderer(env.model, height=480, width=640)

    steps = int(SECONDS / env.control_dt)
    frame_interval = max(1, round(1 / (FPS * env.control_dt)))

    frames = []
    measurements = []
    last_segment = None
    stopped_early = False

    try:
        for step in range(steps):
            t = step * env.control_dt
            command, label = command_at(t)

            env._command[:] = command

            # Recompute observation so policy sees the updated command.
            obs = env._observation()

            if label != last_segment:
                print(f"{t:5.1f}s | {label} | "
                      f"vx={command[0]:.2f}, yaw={command[2]:+.2f}")
                last_segment = label

            if step % frame_interval == 0:
                camera.lookat[:] = env.data.qpos[:3]
                renderer.update_scene(env.data, camera=camera)

                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)

                draw.rectangle((8, 8, 340, 64), fill=(20, 20, 20))
                draw.text(
                    (16, 14),
                    f"Time: {t:.1f}s | {label}",
                    fill="white",
                )
                draw.text(
                    (16, 37),
                    f"vx={command[0]:.2f} m/s  yaw={command[2]:+.2f} rad/s",
                    fill="white",
                )
                frames.append(np.asarray(frame))

            action, _ = policy.predict(
                normalize_observation(obs, mean, var),
                deterministic=True,
            )

            obs, _, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32)
            )

            measurements.append((
                label,
                float(command[2]),
                float(info["angular_vel"][2]),
            ))

            if terminated or truncated:
                print(f"Episode ended at {t:.2f}s "
                      f"(fallen={info.get('fallen')})")
                stopped_early = True
                break

    finally:
        renderer.close()
        env.close()

    if not frames:
        raise RuntimeError("No frames were rendered.")

    print("Encoding MP4...")
    with imageio.get_writer(
        str(OUTPUT),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"\nSaved: {OUTPUT.resolve()}")
    print(f"Video duration: {len(frames)/FPS:.2f}s")
    print(f"Completed full episode: {not stopped_early}")

    print("\nMeasured body-frame yaw rates:")
    for label in ["STRAIGHT", "LEFT TURN", "RIGHT TURN"]:
        values = [m[2] for m in measurements if m[0] == label]
        if values:
            print(f"{label:12s}: mean={np.mean(values):+.3f} rad/s")

if __name__ == "__main__":
    main()
