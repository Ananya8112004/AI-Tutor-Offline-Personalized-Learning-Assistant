import fitz
import os
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def clean_and_chunk(text, chunk_size=512):
    # Remove extra spaces and split into sentences
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.?!])\s+', text)

    # Chunk sentences together
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence.split()) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def save_chunks(chunks, output_file="data/pdf_chunks.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")
    print(f"✅ Saved {len(chunks)} chunks to {output_file}")

if __name__ == "__main__":
    pdf_file = input("🔍 Enter path to PDF file: ").strip()
    if not os.path.exists(pdf_file):
        print("❌ PDF file not found!")
    else:
        raw_text = extract_text_from_pdf(pdf_file)
        chunks = clean_and_chunk(raw_text)
        save_chunks(chunks)
