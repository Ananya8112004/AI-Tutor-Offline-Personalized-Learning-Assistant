from datasets import load_dataset
from transformers import T5Tokenizer
import os

# Paths
tokenizer_name = "t5-small"
jsonl_path = "data/train_data.jsonl"
output_dir = "data/tokenized"

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

# Load dataset from JSONL
dataset = load_dataset("json", data_files={"train": jsonl_path})["train"]

# Tokenization function
def tokenize(example):
    input_enc = tokenizer(example["input"], truncation=True, padding="max_length", max_length=512)
    target_enc = tokenizer(example["output"], truncation=True, padding="max_length", max_length=128)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

# Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True)

# Save to disk
tokenized_dataset.save_to_disk(output_dir)
print(f"✅ Tokenized dataset saved to: {output_dir}")
