from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load final checkpoint
model = T5ForConditionalGeneration.from_pretrained("model_output/checkpoint-7580")
tokenizer = T5Tokenizer.from_pretrained("model_output/checkpoint-7580")

# Save to reusable directory
model.save_pretrained("model/t5_custom")
tokenizer.save_pretrained("model/t5_custom")

print("✅ Model exported to: model/t5_custom")
