


import os
from pypdf import PdfReader


def file_extracter(file_path):

    if file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
        
    elif  file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    

    return text


def chunk_document(text, source, chunk_size=200, overlap=50):

    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "source": source,
            "chunk_id": chunk_id
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def load_pdfs_from_folder(folder_path):
    all_chunks = []

    for filename in os.listdir(folder_path):
            
        
        full_path = os.path.join(folder_path, filename)

        text = file_extracter(full_path)
        chunks = chunk_document(text, filename)

        all_chunks.extend(chunks)

    return all_chunks
