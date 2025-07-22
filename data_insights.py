import pandas as pd
import os

# Replace with your local dataset folder path
dataset_folder = r"data"

# List all CSV files and display basic info
for filename in os.listdir(dataset_folder):
    if filename.endswith(".csv"):
        path = os.path.join(dataset_folder, filename)
        print(f"\n📂 Dataset: {filename}")
        try:
            df = pd.read_csv(path)
            print("✅ Shape:", df.shape)
            print("✅ Columns:", list(df.columns))
            print("📊 Sample rows:\n", df.head(2))
        except Exception as e:
            print("❌ Error reading this file:", e)
