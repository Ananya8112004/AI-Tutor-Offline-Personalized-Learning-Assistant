from transformers import T5Tokenizer, T5ForConditionalGeneration
from evaluate import load as load_metric
import pandas as pd

# Load custom model
tokenizer = T5Tokenizer.from_pretrained("model/t5_custom")
model = T5ForConditionalGeneration.from_pretrained("model/t5_custom")
model.to("cuda")  # Use GPU if available

def predict(text, prefix=""):
    input_text = f"{prefix}: {text}" if prefix else text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    output_ids = model.generate(inputs["input_ids"], max_length=100)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Load metric
rouge = load_metric("rouge")

# Evaluate on a few examples from CNN/DailyMail
df = pd.read_csv("data/cnn_dailymail.csv")

def evaluate_model(test_data, n=10):
    preds, refs = [], []
    for i in range(n):
        input_text = test_data['input'][i]
        true_output = test_data['output'][i]
        pred = predict(input_text)
        preds.append(pred)
        refs.append(true_output)

    results = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    print("\n📊 ROUGE Evaluation:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

# Run it
evaluate_model(df)
