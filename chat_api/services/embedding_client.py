import httpx # A modern, async-friendly HTTP client
from typing import List, Dict, Any, Optional

from chat_api.config.settings import settings
from embedding_worker.models.embedding_model import SearchRequest, SearchResponse # Reusamos modelos del embedding-worker

class EmbeddingServiceClient:
    def __init__(self):
        self.base_url = settings.EMBEDDING_WORKER_URL
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def search_similar_projects(self, query_text: str, project_ids_subset: Optional[List[str]] = None) -> List[str]:
        """
        Sends a query to the embedding worker to find similar projects.
        """
        request_data = SearchRequest(query_text=query_text, project_ids_subset=project_ids_subset or [])
        try:
            response = await self.client.post("/search", json=request_data.model_dump(), timeout=60) # Increased timeout for embedding search
            response.raise_for_status()
            return SearchResponse(**response.json()).similar_project_ids
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error during embedding search: {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"❌ Network error during embedding search: {e}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred during embedding search: {e}")
            return []

    async def get_health(self) -> Dict[str, Any]:
        """Checks the health status of the embedding worker."""
        try:
            response = await self.client.get("/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"❌ Network error checking embedding worker health: {e}")
            return {"status": "UNAVAILABLE", "error": str(e)}
        except Exception as e:
            print(f"❌ An unexpected error occurred while checking embedding worker health: {e}")
            return {"status": "UNKNOWN_ERROR", "error": str(e)}