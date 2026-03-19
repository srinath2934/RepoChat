"""
Embeddings - Convert text to vectors using HuggingFace
"""

import logging

import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import login

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Use CUDA when available, otherwise fallback to CPU."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embeddings_model():
    """
    Initialize embeddings via Hugging Face Inference API.
    Fast, serverless, and doesn't use local CPU/RAM.
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        logger.warning("HUGGINGFACEHUB_API_TOKEN not found. RAG will fail.")
        return None
        
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=token,
    )

    logger.info("✅ HuggingFace Inference API connected")
    return embeddings


if __name__ == "__main__":
    embeddings = get_embeddings_model()

    sample_text = "def hello(): print('Hello, world!')"
    vector = embeddings.embed_query(sample_text)

    print(f"Sample text: {sample_text}")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
