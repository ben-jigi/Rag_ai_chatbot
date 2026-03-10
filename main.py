from fastapi import FastAPI
from pydantic import BaseModel
from Rag_arch import create_add_load, query_index, add_to_index
from document_loader import load_pdfs_from_folder, chunk_document
import os
import requests

app = FastAPI()



OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(prompt):

    payload = {
        "model": "phi3:latest",
        "prompt": prompt,
        "stream": False
    }

    try:
        print("Sending request to Ollama...")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload
        )

        print("Response received")

        data = response.json()

        return data.get("response", "No response generated.")

    except Exception as e:
        print("OLLAMA ERROR:", e)
        return f"Error generating response: {str(e)}"

# ----------- STARTUP LOGIC --------------

folder_path = "data"
chunks = load_pdfs_from_folder(folder_path)
index, chunks = create_add_load(chunks)

chat_history = []

# ----------- REQUEST MODEL --------------

class QueryRequest(BaseModel):
    query: str


class AddRequest(BaseModel):
    text: str


# ----------- CHAT ENDPOINT --------------

@app.post("/chat")
def chat(request: QueryRequest):
    global chat_history, index, chunks

    user_input = request.query

    results = query_index(index, user_input, chunks)

    if not results:
        return {"response": "No relevant information found in the documents."}
    
    context = "\n".join([r["text"] for r in results])

    # Last 3 memory turns
    memory_context = ""
    for turn in chat_history[-3:]:
        memory_context += f"User: {turn['question']}\n"
        memory_context += f"Assistant: {turn['answer']}\n"

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't know.

Conversation History:
{memory_context}

Context:
{context}

Current Question:
{user_input}

Answer:
"""

    answer = generate_answer(prompt)

    chat_history.append({
        "question": user_input,
        "answer": answer
    })

    if len(chat_history) > 10:
        chat_history.pop(0)

    return {"response": answer}


# ----------- ADD KNOWLEDGE ENDPOINT -----------

@app.post("/add")
def add_knowledge(request: AddRequest):
    global index, chunks

    new_chunks = chunk_document(request.text, source="user_input")
    index, chunks = add_to_index(index, new_chunks, chunks)

    return {"message": "Knowledge added successfully"}






