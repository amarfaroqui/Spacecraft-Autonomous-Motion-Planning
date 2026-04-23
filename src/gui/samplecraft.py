import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import joblib
import time
import os
import sys

# Import environment (absolute import)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment import AsteroidEnvironment

# Load trained model
model_path = r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data\rl_model.pkl"
model = joblib.load(model_path)

# Environment setup
env = AsteroidEnvironment(grid_size=10, num_obstacles=15)
state = env.reset()

# GUI setup
root = tk.Tk()
root.title("🚀 Autonomous Spacecraft Mission Control")
root.geometry("900x600")
root.config(bg="black")

# --- Left Simulation Canvas ---
canvas = tk.Canvas(root, width=600, height=600, bg="black", highlightthickness=0)
canvas.pack(side="left", fill="both", expand=True)

# --- Right Control Panel ---
panel = tk.Frame(root, bg="#111111", width=300)
panel.pack(side="right", fill="y")

title = tk.Label(panel, text="Mission Control", font=("Orbitron", 18, "bold"), fg="cyan", bg="#111111")
title.pack(pady=15)

info_text = tk.StringVar()
info_label = tk.Label(panel, textvariable=info_text, font=("Consolas", 12), fg="white", bg="#111111", justify="left")
info_label.pack(pady=20)

# --- Controls ---
btn_frame = tk.Frame(panel, bg="#111111")
btn_frame.pack(pady=20)

def reset_sim():
    global state
    state = env.reset()
    draw_grid(state)
    info_text.set("Mission Reset!\nReady for launch 🚀")

reset_btn = ttk.Button(btn_frame, text="🔁 Reset", command=reset_sim)
reset_btn.grid(row=0, column=0, padx=10)

start_btn = ttk.Button(btn_frame, text="▶️ Start", command=lambda: step_simulation())
start_btn.grid(row=0, column=1, padx=10)

# --- Drawing ---
cell_size = 60
rows, cols = env.grid_size, env.grid_size

def draw_grid(state):
    canvas.delete("all")

    # Draw grid background (space)
    for i in range(rows):
        for j in range(cols):
            x0, y0 = j * cell_size, i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size

            color = "black"
            if (i, j) in env.obstacles:
                canvas.create_oval(x0+15, y0+15, x1-15, y1-15, fill="gray25", outline="gray40")  # asteroid
            elif (i, j) == tuple(env.asteroid_pos):
                canvas.create_oval(x0+10, y0+10, x1-10, y1-10, fill="limegreen", outline="green")
            elif (i, j) == tuple(state):
                canvas.create_oval(x0+15, y0+15, x1-15, y1-15, fill="deepskyblue", outline="cyan")

    # Update mission stats
    dist = np.linalg.norm(np.array(state) - np.array(env.asteroid_pos))
    info_text.set(f"Spacecraft: {state}\nTarget: {env.asteroid_pos}\nObstacles: {len(env.obstacles)}\nDistance: {dist:.2f}")

# --- Simulation Step ---
def step_simulation():
    global state
    draw_grid(state)

    action = model.predict([state])[0]
    next_state, reward, done = env.step(action)
    state = next_state

    draw_grid(state)

    if not done:
        root.after(300, step_simulation)
    else:
        if state == env.asteroid_pos:
            info_text.set("✅ Target Reached!\nMission Successful!")
        else:
            info_text.set("💥 Collision Detected!\nMission Failed!")

draw_grid(state)
info_text.set("System Ready...\nPress ▶️ to begin mission.")

root.mainloop()
