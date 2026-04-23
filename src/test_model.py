import os
import numpy as np
from joblib import load  # use joblib instead of pickle

# Paths
data_folder = r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data"
model_path = os.path.join(data_folder, 'rl_model.pkl')
dataset_path = os.path.join(data_folder, 'rl_dataset.npy')

# Check files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

# Load model using joblib
model = load(model_path)

# Load dataset
dataset = np.load(dataset_path, allow_pickle=True)
states = np.array([s for s, a in dataset])
actions_true = np.array([a for s, a in dataset])

# Predict actions
actions_pred = model.predict(states)

# Compute accuracy
accuracy = np.mean(actions_pred == actions_true) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Optional: display some predictions
print("\nSample predictions:")
for i in range(10):
    print(f"State: {states[i]}, True: {actions_true[i]}, Predicted: {actions_pred[i]}")
