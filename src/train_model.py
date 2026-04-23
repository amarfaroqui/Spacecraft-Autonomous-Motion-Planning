import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = np.load(r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data\rl_dataset.npy", allow_pickle=True)

# Split states and actions
X = np.array([s for s, a in data])  # states
y = np.array([a for s, a in data])  # actions

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

# Save model
import joblib
joblib.dump(model, r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data\rl_model.pkl")
print("Model saved as rl_model.pkl")
