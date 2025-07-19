import os

class Settings:
    """Application settings for the Chat API."""
    
    # URLs of other microservices
    DATA_SERVICE_URL: str = os.getenv("DATA_SERVICE_URL", "http://127.0.0.1:8001")
    EMBEDDING_WORKER_URL: str = os.getenv("EMBEDDING_WORKER_URL", "http://127.0.0.1:8002")

    # Large Language Model (LLM) settings
    # Example for OpenAI API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o") # Or "gpt-3.5-turbo", "gemini-pro", etc.
    OPENAI_LLM_TEMPERATURE: float = 0.7
    
    # Add other LLM specific settings if you choose a different provider (e.g., Google, Anthropic)
    # GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "your_google_api_key_here")
    # GOOGLE_MODEL_NAME: str = os.getenv("GOOGLE_MODEL_NAME", "gemini-pro")


    # Conversation settings
    MAX_CONVERSATION_HISTORY: int = 5 # Number of previous turns to keep in context
    
settings = Settings()