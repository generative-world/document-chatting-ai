import os

# Configuration for embeddings, models, and DB
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
VECTORSTORE_PATH = './db/vectorstore.db'

# Model configuration for LLM (e.g., Mistral 7B, Mixtral 8x7B)
LLM_MODEL_NAME = "mistral-7B"  # Example model name
USE_GPU = True
