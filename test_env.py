
from env import GridWorldEnv

# Create the environment
env = GridWorldEnv()

print("Environment created!")

# Reset the environment
obs, info = env.reset()
print("Reset successful!")
print("Initial Observation:", obs)

# Take 20 random actions
for step in range(20):

    action = env.action_space.sample()

    print(f"\nStep {step + 1}")
    print("Action:", action)

    obs, reward, terminated, truncated, info = env.step(action)

    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    if terminated or truncated:
        print("\nEpisode finished! Resetting...\n")
        obs, info = env.reset()

print("\nEnvironment test completed successfully!")

env.close()