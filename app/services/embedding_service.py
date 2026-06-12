"""
Embedding Service.

Generates vector embeddings for texts using OpenAI's API, integrated with
a caching layer to minimize API calls for identical texts.
"""
from openai import OpenAI

from app.config import settings
from app.services.query_cache_service import query_cache


openai_client = OpenAI(api_key=settings.openai_api_key)

def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """
    Generate dense vector embeddings for a list of strings.
    
    Checks the Redis query cache first. For cache misses, it batches the remaining
    texts to the OpenAI Embeddings API, then caches the new results.
    
    Args:
        texts: A list of text chunks to embed.
        model: The embedding model to use (defaults to settings).
        
    Returns:
        A list of float lists representing the embedding vectors.
    """
    if not texts:
        return []
    if model is None:
        model = settings.embedding_model

    results: list[list[float] | None] = [None] * len(texts)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for i, text in enumerate(texts):
        cached = query_cache.get_embedding(text)
        if cached is not None:
            results[i] = cached

        else:
            miss_indices.append(i)
            miss_texts.append(text)
        
    if miss_texts:
        response = openai_client.embeddings.create(input=miss_texts, model=model)
        for idx_in_misses, item in enumerate(response.data):
            original_idx = miss_indices[idx_in_misses]
            vector = item.embedding
            results[original_idx] = vector
            query_cache.set_embedding(miss_texts[idx_in_misses], vector)

    return [r for r in results if r is not None]

