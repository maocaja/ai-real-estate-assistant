from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from embedding_worker.api.routes import router, _build_index_background_task
from embedding_worker.repository.embedding_repository import EmbeddingRepository
from embedding_worker.config.settings import settings

# Initialize the repository globally so it loads the model on startup
# and can be accessed by the startup event.
embedding_repo_instance = EmbeddingRepository()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    Handles initial FAISS index build and periodic refreshes.
    """
    print("🚀 Embedding Worker starting up...")
    
    # Initial build of the FAISS index at startup
    await _build_index_background_task() # Run initial build directly, not background task

    # Start periodic index refresh in the background
    async def periodic_index_refresh():
        while True:
            await asyncio.sleep(settings.EMBEDDING_REFRESH_INTERVAL_SECONDS)
            print(f"⏰ Initiating periodic index refresh (every {settings.EMBEDDING_REFRESH_INTERVAL_SECONDS} seconds)...")
            await _build_index_background_task()

    # Schedule the periodic task
    periodic_task = asyncio.create_task(periodic_index_refresh())
    
    yield # Application is running

    # Shutdown events
    print("🛬 Embedding Worker shutting down...")
    periodic_task.cancel() # Cancel the periodic task on shutdown
    try:
        await periodic_task # Await to ensure it's cancelled
    except asyncio.CancelledError:
        print("Periodic index refresh task cancelled.")

app = FastAPI(
    title="Embedding Worker - AI Real Estate Assistant",
    description="Microservice for generating and searching project embeddings using FAISS.",
    version="1.0.0",
    lifespan=lifespan # Link the lifespan function here
)

# Include API routes
app.include_router(router)

@app.get("/")
async def read_root():
    """Simple root endpoint for health check."""
    return {"message": "Embedding Worker is running!"}