from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("model/t5_custom")
model = T5ForConditionalGeneration.from_pretrained("model/t5_custom")
# model.to("cuda")

def predict(text, prefix=""):
    input_text = f"{prefix}: {text}" if prefix else text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    # .to("cuda")
    output_ids = model.generate(inputs["input_ids"], max_length=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example: test summarization or QA
# print("🔍 Testing summary...")
# print(predict("summarize: Deep learning models can generate natural language based on input sequences."))
# Input from your cnn_dailymail or scitldr
test_input = "summarize: The Moon is Earth’s only natural satellite. It is about one-sixth as large as Earth. The Moon is held in orbit by Earth’s gravity."

print("🔍 Predicted Summary:")
print(predict(test_input))
