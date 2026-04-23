import numpy as np
import matplotlib.pyplot as plt

class AsteroidEnvironment:
    def __init__(self, grid_size=10, num_obstacles=10):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        # Create empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Place asteroid at center
        self.asteroid_pos = (self.grid_size // 2, self.grid_size // 2)
        self.grid[self.asteroid_pos] = 9  # asteroid marker

        # Randomly place obstacles (avoid asteroid position)
        self.obstacles = []
        for _ in range(self.num_obstacles):
            while True:
                x, y = np.random.randint(0, self.grid_size, size=2)
                if (x, y) != self.asteroid_pos and (x, y) not in self.obstacles:
                    self.obstacles.append((x, y))
                    self.grid[x, y] = -1  # obstacle marker
                    break

        # Random start position (not on asteroid or obstacle)
        while True:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if (x, y) not in self.obstacles and (x, y) != self.asteroid_pos:
                self.agent_pos = (x, y)
                break

        return self.agent_pos

    def get_state(self):
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:   # up
            x = max(0, x - 1)
        elif action == 1: # down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2: # left
            y = max(0, y - 1)
        elif action == 3: # right
            y = min(self.grid_size - 1, y + 1)
        # action 4 = stay still

        new_pos = (x, y)

        # Calculate reward
        reward = -1  # small penalty for each move
        done = False

        if new_pos in self.obstacles:
            reward = -100  # collision
            done = True
        elif new_pos == self.asteroid_pos:
            reward = +100  # reached target
            done = True

        self.agent_pos = new_pos
        return new_pos, reward, done

    def render(self):
        grid_vis = np.copy(self.grid)
        x, y = self.agent_pos
        grid_vis[x, y] = 5  # agent marker
        plt.imshow(grid_vis, cmap='coolwarm')
        plt.title("Asteroid Environment")
        plt.show()
