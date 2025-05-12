from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_from_disk

# Load
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# ✅ This line fixes your issue
model.config.decoder_start_token_id = tokenizer.pad_token_id
model.to("cuda")  # ✅ This line forces model to use GPU

# Load tokenized dataset
dataset = load_from_disk("data/tokenized_split")

training_args = TrainingArguments(
    output_dir="model_output",
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
    metric_for_best_model="loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()
