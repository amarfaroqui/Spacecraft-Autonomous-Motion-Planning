import pandas as pd

# Path to your CSV dataset
csv_path = r"C:\Users\Arjun's\Documents\SEM 5\ML LABS\MINI PROJECT\data\rl_dataset.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Show first 10 rows
print(df.head(10))

# Show basic info about the dataset
print("\nDataset Info:")
print(df.info())

print("\nDataset shape:", df.shape)
