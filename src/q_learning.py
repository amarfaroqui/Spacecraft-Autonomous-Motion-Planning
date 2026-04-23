import numpy as np
from environment import AsteroidEnvironment

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.2, episodes=500):
        self.env = env
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.episodes = episodes
        self.actions = [0, 1, 2, 3, 4]  # up, down, left, right, stay
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(self.actions)))

    def choose_action(self, state):
        x, y = state
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_table[x, y])

    def train(self):
        for ep in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                x, y = state
                nx, ny = next_state
                old_value = self.q_table[x, y, action]
                next_max = np.max(self.q_table[nx, ny])

                # Bellman update
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[x, y, action] = new_value

                state = next_state
                total_reward += reward

            if ep % 50 == 0:
                print(f"Episode {ep}/{self.episodes} - Total Reward: {total_reward}")

        print("Training complete!")

    def test(self):
        state = self.env.reset()
        done = False
        steps = 0
        print("Testing trained agent...\n")
        self.env.render()

        while not done and steps < 50:
            x, y = state
            action = np.argmax(self.q_table[x, y])
            next_state, reward, done = self.env.step(action)
            state = next_state
            self.env.render()
            steps += 1
