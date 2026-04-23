import os
import sys
import numpy as np

sys.path.append(r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\src")

from environment import AsteroidEnvironment
from q_learning import QLearningAgent

data_folder = r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data"
os.makedirs(data_folder, exist_ok=True)

env = AsteroidEnvironment()
agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=500)

# Train the agent
agent.train()

# Generate dataset
dataset = []

for episode in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        dataset.append([list(state), action])  # <- fix here
        state, reward, done = env.step(action)

# Save as .npy
npy_path = os.path.join(data_folder, 'rl_dataset.npy')
np.save(npy_path, np.array(dataset, dtype=object))
print(f"Dataset saved as {npy_path}")

# Save as .csv
csv_path = os.path.join(data_folder, 'rl_dataset.csv')
with open(csv_path, 'w') as f:
    f.write('state,action\n')
    for s, a in dataset:
        f.write(f'"{s}","{a}"\n')
print(f"Dataset saved as {csv_path}")
