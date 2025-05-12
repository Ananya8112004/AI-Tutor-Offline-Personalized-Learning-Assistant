from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load custom model
tokenizer = T5Tokenizer.from_pretrained("model/t5_custom")
model = T5ForConditionalGeneration.from_pretrained("model/t5_custom").to("cuda")  # or "cpu" if no GPU

# Run a sample inference
def run_custom_model(prompt, prefix=""):
    input_text = f"{prefix}: {prompt.strip()}" if prefix else prompt.strip()
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_length=150)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example
print(run_custom_model("Mesophiles grow best in moderate temperature", "generate quiz"))
