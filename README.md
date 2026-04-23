# Spacecraft Autonomous Navigation using Reinforcement Learning

> A Q-Learning based AI agent that navigates a spacecraft through a 2D asteroid field to reach a goal while avoiding obstacles.

---

## Overview

This project demonstrates an **Autonomous Spacecraft Navigation System** using **Reinforcement Learning (Q-Learning)**. The spacecraft learns to navigate through an asteroid field safely, reaching its goal position while avoiding collisions with obstacles.

Developed as part of **UE23CS352A – Machine Learning Mini-Project** at **PES University**.

---

## Problem Statement

Design and implement an AI agent that learns to move a spacecraft through a 2D grid-based asteroid field to reach a goal destination while avoiding obstacles, using Reinforcement Learning.

---

## Approach

The environment is a **10×10 grid world** with:

| Symbol | Represents |
|--------|------------|
| 🔵 Blue | Spacecraft (agent) |
| ⚫ Gray | Asteroids (obstacles) |
| 🟢 Green | Goal (target asteroid) |

The agent learns through **Q-Learning**:
- ✅ Reward for reaching the goal (`+100`)
- ❌ Penalty for collisions (`-100`)
- ➡️ Small penalty per move (`-1`) to encourage efficiency

A **Random Forest Classifier** is then trained on the Q-learning agent's decisions to produce a supervised ML model for deployment in the GUI.

---

## File Structure

```
spacecraft-rl/
│
├── src/
│   ├── environment.py          # Grid world environment (asteroids, spacecraft, goal)
│   ├── q_learning.py           # Q-Learning agent implementation
│   ├── train_agent.py          # RL agent training and evaluation
│   ├── train_model.py          # Random Forest model training and saving
│   ├── test_model.py           # Model testing and accuracy evaluation
│   ├── test_env.py             # Environment visualization and movement testing
│   ├── inspect_npy.py          # Inspect raw .npy dataset entries
│   ├── inspect_dataset.py      # Inspect dataset as a CSV/DataFrame
│   │
│   ├── dataset/
│   │   └── make_dataset.py     # Generate training dataset from trained RL agent
│   │
│   └── gui/
│       ├── spacecraft_gui.py   # Basic Tkinter GUI simulation
│       └── samplecraft.py      # Enhanced Mission Control GUI
│
├── data/                       # (auto-created) Stores model and dataset files
│   ├── rl_dataset.npy
│   ├── rl_dataset.csv
│   └── rl_model.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/spacecraft-rl.git
cd spacecraft-rl
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

Run these steps **in order**:

### Step 1 — Generate Dataset
Train the Q-learning agent and collect state-action data:
```bash
python src/dataset/make_dataset.py
```
Saves `rl_dataset.npy` and `rl_dataset.csv` to the `data/` folder.

### Step 2 — Train the ML Model
Train a Random Forest Classifier on the collected dataset:
```bash
python src/train_model.py
```
Saves `rl_model.pkl` to the `data/` folder.

### Step 3 — Test the Model
Evaluate model accuracy and view sample predictions:
```bash
python src/test_model.py
```

### Step 4 — Visualize the Environment
See the grid world and spacecraft movement:
```bash
python src/test_env.py
```

### Step 5 — Launch the GUI Simulation
Run the spacecraft navigation GUI:
```bash
python src/gui/samplecraft.py
```
Or the simpler GUI:
```bash
python src/gui/spacecraft_gui.py
```

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~84.5% |
| Grid Size | 10 × 10 |
| Obstacles | 15 |
| Training Episodes | 500 |

The agent successfully navigates toward the goal, avoiding obstacles, and learns optimal movement patterns through iterative exploration.

---

## Key Components

### `environment.py`
Implements the `AsteroidEnvironment` class — a grid world with randomized obstacle placement, a fixed goal position, and step-based rewards.

### `q_learning.py`
Implements the `QLearningAgent` using the **Bellman equation** for Q-value updates with configurable learning rate (α), discount factor (γ), and exploration rate (ε).

### `train_model.py`
Trains a `RandomForestClassifier` (scikit-learn) on the Q-agent's state-action pairs and saves the model with `joblib`.

### `samplecraft.py`
A polished **Mission Control GUI** built with Tkinter — displays a live spacecraft simulation with real-time distance tracking and mission status updates.

---

## Dependencies

```
numpy
scipy
pandas
matplotlib
scikit-learn
tensorflow
keras
pymoo
poliastro
seaborn
optuna
