import os
import sys
import tkinter as tk
import numpy as np
import joblib

# Add src folder to Python path
sys.path.append(r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\src")

from environment import AsteroidEnvironment

# Load trained model
model_path = r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data\rl_model.pkl"
model = joblib.load(model_path)

# Environment setup
env = AsteroidEnvironment(grid_size=10, num_obstacles=15)
state = env.reset()

# GUI setup
cell_size = 50
rows, cols = env.grid_size, env.grid_size
root = tk.Tk()
root.title("Autonomous Spacecraft Simulation")
canvas = tk.Canvas(root, width=cols*cell_size, height=rows*cell_size)
canvas.pack()

# Draw grid, agent, goal, and obstacles
def draw_grid(state):
    canvas.delete("all")
    for i in range(rows):
        for j in range(cols):
            x0, y0 = j*cell_size, i*cell_size
            x1, y1 = x0+cell_size, y0+cell_size
            color = "white"
            if (i, j) in env.obstacles:
                color = "gray"
            elif (i, j) == env.asteroid_pos:
                color = "green"
            elif (i, j) == tuple(state):
                color = "blue"
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
    root.update()

# Safe step: avoid obstacles
def safe_step(state, action):
    x, y = state
    # compute next position
    if action == 0:   # up
        x = max(0, x - 1)
    elif action == 1: # down
        x = min(env.grid_size - 1, x + 1)
    elif action == 2: # left
        y = max(0, y - 1)
    elif action == 3: # right
        y = min(env.grid_size - 1, y + 1)
    # action 4 = stay
    new_pos = (x, y)
    if new_pos in env.obstacles:
        # Pick a safe random action instead
        safe_actions = [a for a in range(5) if tuple(simulate_action(state, a)) not in env.obstacles]
        if safe_actions:
            return simulate_action(state, np.random.choice(safe_actions))
    return new_pos

def simulate_action(state, action):
    x, y = state
    if action == 0:   # up
        x = max(0, x - 1)
    elif action == 1: # down
        x = min(env.grid_size - 1, x + 1)
    elif action == 2: # left
        y = max(0, y - 1)
    elif action == 3: # right
        y = min(env.grid_size - 1, y + 1)
    return (x, y)

# Step simulation
done = False
def step_simulation():
    global state, done
    if not done:
        # Predict action using ML model
        action = model.predict([state])[0]
        # Safe next state
        new_state = safe_step(state, action)
        # Determine reward and done
        if new_state in env.obstacles:
            reward = -100
            done = True
        elif new_state == env.asteroid_pos:
            reward = 100
            done = True
        else:
            reward = -1
            done = False
        state = new_state
        draw_grid(state)
        if not done:
            root.after(300, step_simulation)
        else:
            print("Simulation finished!")

# Start simulation
draw_grid(state)
root.after(1000, step_simulation)
root.mainloop()
