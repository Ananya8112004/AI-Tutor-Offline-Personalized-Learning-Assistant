import streamlit as st
st.set_page_config(page_title="PDF Tutor AI", layout="wide")

import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import sys

# Optional: Patch torch.classes to avoid Streamlit conflict
class DummyModule:
    def __getattr__(self, name):
        raise AttributeError(f"torch.classes.{name} is not available.")
import types
sys.modules["torch.classes"] = DummyModule()

# Load model + tokenizer
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to("cuda")
    return tokenizer, model

tokenizer, model = load_model()

# Extract text from uploaded PDF
def extract_text_from_pdf(uploaded_file):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page in pdf_doc:
        full_text += page.get_text()
    return full_text

# Break text into chunks
def chunk_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Run the model task
def run_t5_task(prompt, prefix=""):
    input_text = f"{prefix}: {prompt.strip()}" if prefix else prompt.strip()
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_length=256)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Get most relevant PDF chunks for a question
def get_top_k_chunks(question, chunks, k=3):
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    vectors = vectorizer.transform(chunks + [question])
    similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

# Extract MCQ question and options from model output
def parse_mcq_output(text):
    try:
        question_match = re.search(r'Question:\s*(.*)', text)
        options_match = re.findall(r'([A-D])\.\s*(.*)', text)
        answer_match = re.search(r'Answer:\s*([A-D])', text)

        question = question_match.group(1).strip() if question_match else "N/A"
        options = {opt[0]: opt[1].strip() for opt in options_match}
        answer = answer_match.group(1).strip() if answer_match else "N/A"

        return question, options, answer
    except Exception:
        return "Parse Error", {}, "N/A"

# UI begins
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    st.success("✅ PDF uploaded. Extracting content...")
    text = extract_text_from_pdf(uploaded_pdf)
    chunks = chunk_text(text)
    st.info(f"Document split into {len(chunks)} chunks.")

    mode = st.radio("Choose a task", ["Summarize", "Generate Quiz Question", "Ask Question About PDF"])

    if mode == "Summarize":
        summaries = []
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(f"### 🔹 Chunk {i+1}")
            with st.spinner("Summarizing..."):
                summary = run_t5_task(chunk, "summarize")
            summaries.append(f"Chunk {i+1}:\n{summary}\n")
            st.success(summary)
        if summaries:
            st.download_button("📥 Download All Summaries", "\n\n".join(summaries), file_name="summaries.txt")

    elif mode == "Generate Quiz Question":
        quizzes = []
        for i, chunk in enumerate(chunks[:5]):
            st.markdown(f"### 🧠 Quiz from Chunk {i+1}")
            prompt = (
                "Based on the following passage, create one multiple-choice question with 4 answer options. "
                "Indicate the correct answer clearly.\n\n"
                f"Passage:\n{chunk}\n\n"
                "Format:\nQuestion: <text>\nA. <text>\nB. <text>\nC. <text>\nD. <text>\nAnswer: <A/B/C/D>"
            )
            with st.spinner("Generating quiz..."):
                raw_quiz = run_t5_task(prompt)
                question, options, answer = parse_mcq_output(raw_quiz)
                quizzes.append({
                    "chunk": i+1,
                    "question": question,
                    "options": options,
                    "answer": answer
                })

                st.markdown(f"**Q: {question}**")
                for key in ["A", "B", "C", "D"]:
                    if key in options:
                        st.markdown(f"- {key}. {options[key]}")
                st.markdown(f"✅ **Answer:** {answer}")
        if quizzes:
            st.download_button("📥 Download Quiz as JSON", json.dumps(quizzes, indent=2), file_name="quiz_mcq.json")

    elif mode == "Ask Question About PDF":
        user_question = st.text_input("Enter your question about the PDF:")
        if user_question:
            top_chunks = get_top_k_chunks(user_question, chunks)
            context = " ".join(top_chunks)
            qa_prompt = f"question: {user_question} context: {context}"
            with st.spinner("Thinking..."):
                answer = run_t5_task(qa_prompt)
            st.success(f"Answer: {answer}")
