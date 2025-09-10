from sentence_transformers import SentenceTransformer
import numpy as np

dense_model = SentenceTransformer("BAAI/bge-m3")

def embed_dense(text: str):
    embedding = dense_model.encode(text)
    return (embedding / np.linalg.norm(embedding)).tolist()

def embed_dense_batch(texts, batch_size=32):
    embeddings = dense_model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return normed.tolist()
