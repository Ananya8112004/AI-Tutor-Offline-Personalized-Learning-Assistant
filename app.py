import streamlit as st
st.set_page_config(page_title="PDF Tutor AI", layout="wide")

# Now import anything else
import fitz  # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

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

def get_top_k_chunks(question, chunks, k=3):
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    vectors = vectorizer.transform(chunks + [question])
    similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]
    top_indices = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]


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
            with st.spinner("Generating quiz..."):
                quiz = run_t5_task(chunk, "generate quiz")
            quizzes.append({"chunk": i+1, "quiz": quiz})
            st.warning(quiz)
        if quizzes:
            quiz_text = json.dumps(quizzes, indent=2)
            st.download_button("📥 Download Quiz as JSON", quiz_text, file_name="quiz.json")
    
    elif mode == "Ask Question About PDF":
        user_question = st.text_input("Enter your question about the PDF:")
        if user_question:
            top_chunks = get_top_k_chunks(user_question, chunks)
            context = " ".join(top_chunks) # use first few chunks as context
            qa_prompt = f"question: {user_question} context: {context}"
            with st.spinner("Thinking..."):
                answer = run_t5_task(qa_prompt, "")
            st.success(f"Answer: {answer}")
