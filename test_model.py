import os
import time

from stable_baselines3 import PPO
from env import GridWorldEnv


# This must match the path train.py actually saves to.
# train.py does: FINAL_MODEL_PATH = "ppo_star_final"; model.save(FINAL_MODEL_PATH)
# which writes "ppo_star_final.zip" in the current directory (no "models/" folder).
MODEL_PATH = "ppo_star_final"

if not os.path.exists(MODEL_PATH + ".zip"):
    raise FileNotFoundError(
        f"Could not find trained model at '{MODEL_PATH}.zip'. "
        f"Run train.py first to produce it."
    )

# Create environment with rendering enabled for visual testing
env = GridWorldEnv(render_mode="human")

# Load trained model
model = PPO.load(MODEL_PATH)

obs, info = env.reset()

episode = 1

while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    print(
        f"Episode: {episode} | "
        f"Action: {action} | "
        f"Reward: {reward:.2f} | "
        f"Stars Left: {info['stars_left']} | "
        f"Steps: {info['steps']}"
    )

    time.sleep(0.2)

    if terminated:

        print("\n===================")
        print("EPISODE TERMINATED")
        print("===================\n")

        obs, info = env.reset()
        episode += 1

    elif truncated:

        print("\n===================")
        print("EPISODE TRUNCATED")
        print("===================\n")

        obs, info = env.reset()
        episode += 1