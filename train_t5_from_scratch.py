from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_from_disk
from evaluate import load as load_metric
import torch
import os

# === Utility to auto-increment folder names ===
def get_next_free_path(base_path):
    if not os.path.exists(base_path):
        return base_path
    i = 1
    while True:
        new_path = f"{base_path}{i}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

# === Load tokenizer and model ===
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Fix required decoder start token
model.config.decoder_start_token_id = tokenizer.pad_token_id
model.to("cuda")

# === Load tokenized train/test dataset ===
dataset = load_from_disk("data/tokenized_split")

# === Load ROUGE metric ===
rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {
        "rouge1": result["rouge1"].mid.fmeasure,
        "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure,
        "rougeLsum": result["rougeLsum"].mid.fmeasure,
    }

# === Setup paths ===
model_output_path = get_next_free_path("model_output")
custom_model_path = get_next_free_path("model/t5_custom")

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=model_output_path,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_dir="logs",
    logging_steps=100,
    save_steps=500,
    eval_steps=500,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    save_total_limit=2
)

# === Create Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === Train ===
trainer.train()

# === Save model ===
os.makedirs(custom_model_path, exist_ok=True)
model.save_pretrained(custom_model_path)
tokenizer.save_pretrained(custom_model_path)

# === Evaluate final metrics ===
final_metrics = trainer.evaluate()
print("\n📊 Final Evaluation Metrics:")
for key, value in final_metrics.items():
    print(f"{key}: {value:.4f}")
