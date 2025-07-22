from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model
model_name = "t5-small"  # You can switch to 't5-base' if RAM/GPU allows
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()

# Load PDF chunks
with open("data/pdf_chunks.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

# Choose task
TASK = "summarize"  # or "quiz"

for i, chunk in enumerate(chunks[:10]):  # test on first 10 chunks for now
    if TASK == "summarize":
        input_text = "summarize: " + chunk
    else:
        input_text = "generate quiz: " + chunk

    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    summary_ids = model.generate(inputs.input_ids, max_length=150, num_beams=4, early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"\n--- Chunk {i + 1} ---")
    print(f"Input:\n{chunk[:300]}...")
    print(f"Output:\n{output}")
