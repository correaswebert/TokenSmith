"""
reranker.py

This module supports re-ranking strategies applied before the generative LLM call.
"""

from typing import Dict, List
import numpy as np
from sentence_transformers import CrossEncoder

from src.embedder import CachedEmbedder

# -------------------------- Cross-Encoder Cache --------------------------
_CROSS_ENCODER_CACHE: Dict[str, CrossEncoder] = {}

def get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
    """
    Fetch the cached cross-encoder model to prevent reloading on every query.
    """
    if model_name not in _CROSS_ENCODER_CACHE:
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


# -------------------------- MMR Cache --------------------------
_MMR_CACHE: Dict[str, CrossEncoder] = {}

def get_mmr(model_name: str = "Qwen3-Embedding-4B-Q5_K_M.gguf"):
    """
    Fetch the cached mmr model to prevent reloading on every query.
    """
    if model_name not in _MMR_CACHE:
        _MMR_CACHE[model_name] = CachedEmbedder(model_name)
    return _MMR_CACHE[model_name]


# -------------------------- Reranking Strategies -------------------------
def rerank_with_cross_encoder(query: str, chunks: List[str], top_n: int) -> List[str]:
    """
    Reranks a list of documents using the cross-encoder model.
    """
    if not chunks:
        return []

    model = get_cross_encoder()

    # Create pairs of [query, chunk] for the model
    pairs = [(query, chunk) for chunk in chunks]

    # Predict the scores
    scores = model.predict(pairs, show_progress_bar=False)

    # Combine chunks with their scores and sort
    chunk_with_scores = list(zip(chunks, scores))
    chunk_with_scores.sort(key=lambda x: x[1], reverse=True)

    reordered_chunks = []

    for chunk, score in chunk_with_scores:
        # Only include chunks with positive scores
        if score > 0:
            reordered_chunks.append(chunk)

    # Return top N chunks
    return reordered_chunks[0:top_n]


# -------------------------- MMR Reranking --------------------------------
def rerank_with_mmr(
    query: str,
    chunks: List[str],
    top_n: int,
    lambda_param: float = 0.7,
) -> List[str]:
    """
    Reranks chunks using Maximal Marginal Relevance (MMR).

    Iteratively selects the chunk that best balances:
      - Relevance to the query        (weighted by lambda_param)
      - Dissimilarity to already-selected chunks  (weighted by 1 - lambda_param)
    """
    if not chunks:
        return []

    top_n = min(top_n, len(chunks))

    # Embed query and all chunks once upfront
    query_embedding = np.array(get_embedding(query))
    chunk_embeddings = np.array([get_embedding(chunk) for chunk in chunks])

    # Cosine similarity helper
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # Relevance scores: similarity of each chunk to the query
    relevance_scores = [cosine(emb, query_embedding) for emb in chunk_embeddings]

    selected_indices: List[int] = []
    candidate_indices = list(range(len(chunks)))

    while len(selected_indices) < top_n and candidate_indices:
        best_index = None
        best_score = float("-inf")

        for idx in candidate_indices:
            relevance = relevance_scores[idx]

            # Redundancy: max similarity to any already-selected chunk
            if selected_indices:
                redundancy = max(
                    cosine(chunk_embeddings[idx], chunk_embeddings[sel])
                    for sel in selected_indices
                )
            else:
                redundancy = 0.0  # nothing selected yet, no redundancy penalty

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = idx

        selected_indices.append(best_index)
        candidate_indices.remove(best_index)

    return [chunks[i] for i in selected_indices]


# -------------------------- Reranking Router -----------------------------
def rerank(query: str, chunks: List[str], mode: str, top_n: int) -> List[str]:
    """
    Routes to the appropriate reranker based on the mode in the config.
    """
    if mode == "cross_encoder":
        return rerank_with_cross_encoder(query, chunks, top_n)
        
    elif mode == "mmr":
        filtered_chunks = rerank_with_cross_encoder(query, chunks, top_n * 2)
        return rerank_with_mmr(query, filtered_chunks, top_n)

    return chunks