# Custom Tutor AI Project Documentation

## Overview

This project aims to build a fully offline, custom-trained Tutor AI model that explains academic concepts, answers university-level questions, and summarizes PDF or slide content. The model is trained **from scratch** using custom datasets on a Windows laptop with limited GPU resources (RTX 3050, 16GB RAM). The final model will be deployed in a UI inspired by the GitHub project [AI\_Tutor](https://github.com/098765d/AI_Tutor).

---

## System Requirements

### Hardware:

* ASUS VivoBook Pro 15 OLED M6500
* GPU: NVIDIA RTX 3050 (4–6 GB VRAM)
* RAM: 16 GB

### Software:

* OS: Windows 10/11
* Python 3.10+
* PyTorch with CUDA
* HuggingFace Transformers & Datasets
* Streamlit (for UI)
* Optional: WSL2 (for better training compatibility)

---

## Project Structure

```plaintext
C:\Users\ashis\Ashish\coding\github\AI_Tutor\dataset\data\
│   cnn_dailymail.csv      ← Summarization
│   sciq_quiz.csv          ← Quiz generation (MCQs)
│   scitldr.csv            ← Scientific summarization
│   squad_qa.csv           ← Question Answering
```

---

## Model Architecture

### Target:

* Type: Decoder-only Transformer (GPT-style)
* Params: \~100–150M
* Layers: 12
* Hidden size: 768
* Heads: 12
* Sequence Length: 1024 tokens
* Dropout: 0.1 (optional)

### Libraries:

* HuggingFace Transformers (GPT2LMHeadModel)

---

## Dataset Plan

### Phase 1: Pretraining (Unsupervised Language Modeling)

* **Wikipedia (5–10GB)**
* **Project Gutenberg (Books)**
* **ArXiv abstracts / scientific papers**
* **StackExchange (academic Q\&A)**

### Phase 2: Fine-Tuning (Instruction Supervised Learning)

* **Dolly 15k**: Instruction-following
* **ELI5**: Explanatory Q\&A
* **SciTLDR**: Scientific summarization
* **CNN/DailyMail**: News summarization (optional)

### Tools:

* `datasets` (HuggingFace)
* `wikiextractor` (for Wikipedia)
* `sentencepiece` or `ByteLevelBPETokenizer` for tokenizer

---

## Training Strategy

### Phase 1: Pretraining

* **Objective:** Causal Language Modeling (CLM)
* **Steps:** 100k–300k
* **Batch Size:** 4 (accumulate gradients to simulate 32)
* **Learning Rate:** 2e-4 (cosine decay)
* **FP16:** Enabled
* **Gradient Checkpointing:** Enabled

### Phase 2: Fine-Tuning

* **Objective:** Prompt -> Response supervision
* **Dataset:** ELI5 + Dolly15k + summarization tasks
* **Epochs:** 3–5
* **Learning Rate:** 5e-5

---

## UI Integration (AI\_Tutor-Based)

### Adjustments to AI\_Tutor:

1. **Replace OpenAI Assistant API calls with local model inference.**
2. **Integrate PyTorch-based model loader.**
3. **Ensure PDF upload works using `PyMuPDF` or `pdfplumber`**.
4. **Adjust chat backend to call `model.generate(...)` instead of API**.
5. **Maintain chat memory using a simple list in `session_state`.**

### Streamlit Flow:

* Upload PDF
* Extract and display text
* Enter query about text
* Model responds with explanation

### Example `app.py` Logic:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("model/checkpoints/finetune", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

st.title("Custom Tutor AI")
user_input = st.text_input("Ask your academic question:")
if user_input:
    prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output_ids = model.generate(input_ids, max_length=512, top_p=0.9, temperature=0.8)
    answer = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    st.write(answer)
```

---

## Deployment Instructions

1. Clone or create project structure
2. Install requirements: `pip install -r requirements.txt`
3. Train model:

   * Run `python training/pretrain.py`
   * Run `python training/finetune.py`
4. Launch UI: `streamlit run app/app.py`
5. Interact with your fully offline tutor AI!

---

## Future Enhancements

* Add conversation memory
* Support PDF chunking and multi-turn Q\&A
* Use LoRA for continued instruction tuning
* Train larger model (300M+) with mixed precision and DeepSpeed
* Add unit tests and inference benchmarks

---

## Credits

* HuggingFace for Transformers and Datasets
* Databricks (Dolly 15k)
* ELI5, SciTLDR, Wikipedia, ArXiv – for training data
* UI Template: [AI\_Tutor](https://github.com/098765d/AI_Tutor)
