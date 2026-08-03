import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from env import GridWorldEnv


# ==========================================
# Config
# ==========================================

TOTAL_TIMESTEPS = 100000           # how many steps to train for this run
FINAL_MODEL_PATH = "ppo_star_final"
CSV_LOG_DIR = "./logs/csv/"


def main():

    # ==========================================
    # Create folders
    # ==========================================

    os.makedirs("models", exist_ok=True)
    os.makedirs(CSV_LOG_DIR, exist_ok=True)

    # ==========================================
    # Environment
    # ==========================================

    env = GridWorldEnv()
    env = Monitor(env)

    # ==========================================
    # PPO Model (fresh start or resume)
    # ==========================================

    if os.path.exists(FINAL_MODEL_PATH + ".zip"):

        print(f"\nFound existing model at {FINAL_MODEL_PATH}.zip — resuming training.\n")

        model = PPO.load(FINAL_MODEL_PATH, env=env)

    else:

        print("\nNo existing model found — starting fresh training run.\n")

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            verbose=1
        )

    # ==========================================
    # CSV Logging
    # ==========================================
    # Every SB3 training metric (reward, episode length, value loss,
    # entropy, KL, clip fraction, etc.) is written to progress.csv in
    # this folder after every rollout. Each run gets its own numbered
    # subfolder so earlier runs' CSVs are never overwritten.

    existing_runs = [
        d for d in os.listdir(CSV_LOG_DIR)
        if os.path.isdir(os.path.join(CSV_LOG_DIR, d)) and d.startswith("run_")
    ]

    run_number = len(existing_runs) + 1
    csv_run_path = os.path.join(CSV_LOG_DIR, f"run_{run_number}")

    new_logger = configure(csv_run_path, ["stdout", "csv"])
    model.set_logger(new_logger)

    print(f"Logging this run's metrics to: {csv_run_path}/progress.csv\n")

    # ==========================================
    # Train
    # ==========================================

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=False
    )

    # ==========================================
    # Save final model
    # ==========================================

    model.save(FINAL_MODEL_PATH)

    print("\nTraining complete! Model saved to", FINAL_MODEL_PATH + ".zip")
    print("Metrics saved to", csv_run_path + "/progress.csv")
    print("Run this script again to keep training from where it left off.\n")

    env.close()


if __name__ == "__main__":
    main()