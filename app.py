import streamlit as st
st.set_page_config(page_title="PDF Tutor AI", layout="wide")

# Now import anything else
import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load model + tokenizer (T5)
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small").to("cuda")
    return tokenizer, model

tokenizer, model = load_model()

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in pdf_doc:
        full_text += page.get_text()
    return full_text

# Chunk text into ~512-token chunks
def chunk_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Generate output from model
def run_t5_task(prompt, prefix):
    input_text = f"{prefix}: {prompt.strip()}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_length=150)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)



uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    st.success("✅ PDF uploaded. Extracting content...")
    text = extract_text_from_pdf(uploaded_pdf)
    chunks = chunk_text(text)
    st.info(f"Document split into {len(chunks)} chunks.")

    mode = st.radio("Choose a task", ["Summarize", "Generate Quiz Question", "Ask Question About PDF"])

    if mode == "Summarize":
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(f"### 🔹 Chunk {i+1}")
            with st.spinner("Summarizing..."):
                summary = run_t5_task(chunk, "summarize")
            st.success(summary)

    elif mode == "Generate Quiz Question":
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(f"### 🧠 Quiz from Chunk {i+1}")
            with st.spinner("Generating quiz..."):
                quiz = run_t5_task(chunk, "generate quiz")
            st.warning(quiz)

    elif mode == "Ask Question About PDF":
        user_question = st.text_input("Enter your question about the PDF:")
        if user_question:
            context = " ".join(chunks[:3])  # use first few chunks as context
            qa_prompt = f"question: {user_question} context: {context}"
            with st.spinner("Thinking..."):
                answer = run_t5_task(qa_prompt, "")
            st.success(f"Answer: {answer}")
