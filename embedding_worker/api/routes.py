from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any

from embedding_worker.repository.embedding_repository import EmbeddingRepository
from embedding_worker.models.embedding_model import SearchRequest, SearchResponse

router = APIRouter()
embedding_repo = EmbeddingRepository() # Initialize the embedding repository

# Function to be run in the background to build/refresh the index
async def _build_index_background_task():
    """
    Background task to fetch data and build the FAISS index.
    """
    try:
        print("Initiating background task to build FAISS index...")
        projects_data = embedding_repo.fetch_data_for_indexing()
        embedding_repo.build_faiss_index(projects_data)
        print("✅ FAISS index built/refreshed successfully in background.")
    except Exception as e:
        print(f"❌ Error during background FAISS index build/refresh: {e}")
        # In a real app, you might want more sophisticated error reporting
        # or retry mechanisms here.

@router.post("/search", response_model=SearchResponse)
async def search_projects(request: SearchRequest):
    """
    Performs a semantic similarity search based on a query text.
    Optionally, the search can be limited to a subset of project IDs.
    """
    if not embedding_repo.is_model_loaded():
        raise HTTPException(status_code=503, detail="Embedding model not loaded. Service is not ready.")
    if not embedding_repo.is_index_built():
        raise HTTPException(status_code=503, detail="FAISS index not built. Service is not ready. Please wait or trigger /rebuild-index if needed.")

    try:
        similar_ids = embedding_repo.search_similar_projects(
            query_text=request.query_text,
            project_ids_subset=request.project_ids_subset
        )
        return SearchResponse(similar_project_ids=similar_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")

@router.get("/health")
async def health_check():
    """
    Provides a health check for the embedding worker,
    indicating if the model is loaded and the index is built.
    """
    model_loaded = embedding_repo.is_model_loaded()
    index_built = embedding_repo.is_index_built()

    status = "OK"
    if not model_loaded:
        status = "Degraded (Model not loaded)"
    elif not index_built:
        status = "Degraded (Index not built)"

    return {
        "status": status,
        "model_loaded": model_loaded,
        "faiss_index_built": index_built,
        "indexed_vector_count": embedding_repo._faiss_index.ntotal if embedding_repo.is_index_built() else 0
    }

@router.post("/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Triggers a rebuild of the FAISS index in the background.
    Useful for refreshing data without blocking the API.
    """
    if not embedding_repo.is_model_loaded():
        raise HTTPException(status_code=503, detail="Embedding model not loaded. Cannot rebuild index.")

    background_tasks.add_task(_build_index_background_task)
    return {"message": "FAISS index rebuild initiated in the background. Check /health for status."}