from datasets import load_dataset
import pandas as pd
import os

# Create data folder
os.makedirs("data", exist_ok=True)

# 1. CNN/DailyMail (Summarization)
print("Downloading CNN/DailyMail...")
cnn = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")  # sample 10k examples
cnn_df = pd.DataFrame({
    "input": ["summarize: " + a for a in cnn["article"]],
    "output": cnn["highlights"]
})
cnn_df.to_csv("data/cnn_dailymail.csv", index=False)

# 2. SciTLDR (Scientific summaries)
print("Downloading SciTLDR...")
sci = load_dataset("allenai/scitldr", split="train", trust_remote_code=True)
sci_df = pd.DataFrame({
    "input": ["summarize: " + " ".join(a) for a in sci["source"]],
    "output": [s[0] if isinstance(s, list) else s for s in sci["target"]]
})
sci_df.to_csv("data/scitldr.csv", index=False)

# 3. SciQ (Quiz / MCQs)
print("Downloading SciQ...")
sciq = load_dataset("sciq", split="train")
sciq_df = pd.DataFrame({
    "input": ["generate quiz: " + p for p in sciq["support"]],
    "question": sciq["question"],
    "correct": sciq["correct_answer"],
    "distractor_1": sciq["distractor1"],
    "distractor_2": sciq["distractor2"],
    "distractor_3": sciq["distractor3"],
})
sciq_df.to_csv("data/sciq_quiz.csv", index=False)

# 4. SQuAD v2 (QA)
print("Downloading SQuAD v2...")
squad = load_dataset("squad_v2", split="train[:10000]")  # 10k examples
squad_df = pd.DataFrame({
    "input": [f"question: {q} context: {c}" for q, c in zip(squad["question"], squad["context"])],
    "output": squad["answers"]
})
squad_df.to_csv("data/squad_qa.csv", index=False)

print("✅ All datasets downloaded and saved to /data folder.")
