from datasets import load_from_disk
from datasets import DatasetDict

# Load the flat tokenized dataset
dataset = load_from_disk("data/tokenized")

# Split into train and validation (5% for validation)
split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

# Save to disk
split_dataset.save_to_disk("data/tokenized_split")

print("✅ Dataset split and saved to data/tokenized_split")
print(f"Train size: {len(split_dataset['train'])}")
print(f"Validation size: {len(split_dataset['test'])}")
