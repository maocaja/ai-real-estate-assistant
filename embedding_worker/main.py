# embedding_worker/main.py

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncio
from dotenv import load_dotenv
load_dotenv()

from embedding_worker.config.settings import settings
from embedding_worker.services.embedding_service import EmbeddingService # Importar el nuevo servicio
from embedding_worker.models.embedding_model import SearchRequest, SearchResponse # Modelos para la API


# Instancia global del EmbeddingService
embedding_service_instance = EmbeddingService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Initializes the embedding model and FAISS index.
    """
    print("🚀 Embedding Worker starting up...")
    
    # Initialize the embedding service (load model and build index)
    await embedding_service_instance.initialize()

    yield # Application is running

    # Shutdown events (none specific needed for embedding_worker for now)
    print("🛬 Embedding Worker shutting down...")

app = FastAPI(
    title="Embedding Worker",
    description="Microservice for generating embeddings and performing vector search.",
    version="1.0.0",
    lifespan=lifespan # Link the lifespan function here
)

@app.get("/")
async def read_root():
    """Simple root endpoint for health check."""
    return {"message": "Embedding Worker is running!"}

@app.post("/search", response_model=SearchResponse)
async def search_projects_endpoint(request: SearchRequest):
    """
    Endpoint to search for semantically similar projects.
    """
    if not embedding_service_instance.is_index_ready():
        raise HTTPException(status_code=503, detail="Embedding index is not yet built. Please wait.")
    
    similar_ids = await embedding_service_instance.search_similar_projects(
        query_text=request.query_text,
        limit=request.limit # Assuming SearchRequest has a limit field, otherwise default to 5
    )
    return SearchResponse(similar_project_ids=similar_ids)

@app.get("/health")
async def health_check():
    """
    Provides a health check for the Embedding Worker, including FAISS index status.
    """
    status = "OK"
    error = None
    faiss_index_built = embedding_service_instance.is_index_ready()

    if not embedding_service_instance.get_model():
        status = "Degraded"
        error = "Embedding model not loaded."
    elif not faiss_index_built:
        status = "Degraded"
        error = "FAISS index not built."

    return {
        "status": status,
        "message": "Embedding Worker is operational.",
        "faiss_index_built": faiss_index_built,
        "error": error
    }