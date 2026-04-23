from environment import AsteroidEnvironment

env = AsteroidEnvironment(grid_size=10, num_obstacles=15)
state = env.reset()
print("Start state:", state)
env.render()

for i in range(10):
    new_state, reward, done = env.step(action=i % 5)
    print(f"Step {i}: State={new_state}, Reward={reward}, Done={done}")
    env.render()
    if done:
        break
