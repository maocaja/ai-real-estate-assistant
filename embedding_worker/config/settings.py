import os

class Settings:
    """Application settings for the Embedding Worker."""
    # URL of the Data Service
    DATA_SERVICE_URL: str = os.getenv("DATA_SERVICE_URL", "http://127.0.0.1:8001")

    # Hugging Face model name for embeddings (e.g., 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    # This model is good for multilingual text, including Spanish.
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Path for a local cache of the embedding model (optional)
    MODEL_CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "embedding_models")

    # Interval (in seconds) to periodically refresh embeddings from data service
    EMBEDDING_REFRESH_INTERVAL_SECONDS: int = 3600 # Every hour

settings = Settings()