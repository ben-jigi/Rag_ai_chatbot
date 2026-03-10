import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle



embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
def embed_text(texts):
    embeddings=embedding_model.encode(texts)
    embeddings=np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings

def create_add_load(chunks, index_path="faiss_index.index", chunks_path="chunks.pkl"):

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        print("Loaded existing FAISS index.")
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        print("Loaded existing FAISS index and chunks.")
        return index, chunks

    else:
        texts = [chunk["text"] for chunk in chunks]
        embeddings=embed_text(texts)

        dim = embeddings.shape[1]
        index=faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        
        with open(chunks_path, "wb") as f:
           pickle.dump(chunks, f)

        
    return index, chunks

    

def add_to_index(index, new_chunks, chunks_list, index_path="faiss_index.index",chunks_path="chunks.pkl"):
    if isinstance(new_chunks, dict):
        new_chunks = [new_chunks]

    texts = [chunk["text"] for chunk in new_chunks]


    new_embeddings = embed_text(texts)
    index.add(new_embeddings)
    chunks_list.extend(new_chunks)
    faiss.write_index(index, index_path)
    
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks_list, f)
    print("new texts has been added")

    return index, chunks_list

def query_index(index, query, chunks_list, top_k=5, threshold=None):
    new_query=embed_text([query])
    similarities, indices= index.search(new_query, top_k)
    top_scores = similarities[0]
    top_indices = indices[0]

    results = []

    for score, idx in zip(top_scores, top_indices):
        if threshold is None or score >= threshold:
            chunk = chunks_list[idx]
            results.append({
               "text": chunk["text"],
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "score": float(score)
            })
    return results



