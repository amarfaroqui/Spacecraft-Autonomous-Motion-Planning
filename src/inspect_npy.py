import numpy as np

# Path to your NumPy dataset
npy_path = r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data\rl_dataset.npy"

# Load the dataset
data = np.load(npy_path, allow_pickle=True)

# Show first 10 entries
for i, (state, action) in enumerate(data[:10]):
    print(f"{i}: State={state}, Action={action}")

print("\nTotal entries:", len(data))
print("Type of first state:", type(data[0][0]))
print("Type of first action:", type(data[0][1]))
