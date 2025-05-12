import pandas as pd
import json
import os

input_dir = "data"
output_file = "data/train_data.jsonl"
records = []

# CNN/DailyMail
df_cnn = pd.read_csv(f"{input_dir}/cnn_dailymail.csv")
for _, row in df_cnn.iterrows():
    records.append({"input": row["input"], "output": row["output"]})

# SciTLDR
df_sci = pd.read_csv(f"{input_dir}/scitldr.csv")
for _, row in df_sci.iterrows():
    records.append({"input": row["input"], "output": row["output"]})

# SciQ
df_sciq = pd.read_csv(f"{input_dir}/sciq_quiz.csv")
for _, row in df_sciq.iterrows():
    q = row["question"]
    options = f"A. {row['correct']} B. {row['distractor_1']} C. {row['distractor_2']} D. {row['distractor_3']}"
    answer = f"Answer: {row['correct']}"
    records.append({
        "input": row["input"],
        "output": f"Question: {q}\nOptions:\n{options}\n{answer}"
    })

# SQuAD
df_squad = pd.read_csv(f"{input_dir}/squad_qa.csv")
for _, row in df_squad.iterrows():
    try:
        # Parse JSON string
        answer_text = eval(row["output"])["text"][0]
        records.append({"input": row["input"], "output": answer_text})
    except Exception:
        continue

# Save all to JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for r in records:
        json.dump(r, f)
        f.write("\n")

print(f"✅ Saved {len(records)} records to {output_file}")
