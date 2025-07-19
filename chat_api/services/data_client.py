import httpx # A modern, async-friendly HTTP client
from typing import List, Dict, Any, Optional

from chat_api.config.settings import settings
from data_service.models.project import Project, ProjectAmenitiesData # Reusamos modelos del data-service

class DataServiceClient:
    def __init__(self):
        self.base_url = settings.DATA_SERVICE_URL
        # Usamos httpx.AsyncClient para operaciones asíncronas
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def get_filtered_projects(
        self,
        city: Optional[str] = None,
        property_type: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_bedrooms: Optional[int] = None,
        min_bathrooms: Optional[int] = None,
        min_area_sqm: Optional[float] = None,
        recommended_use: Optional[str] = None,
        project_name: Optional[str] = None
    ) -> List[Project]:
        """Fetches projects from the data-service with applied filters."""
        params = {k: v for k, v in locals().items() if k not in ['self', 'client'] and v is not None}
        try:
            response = await self.client.get("/projects", params=params, timeout=30)
            response.raise_for_status()
            # Validar la respuesta con el modelo Project del data-service
            return [Project(**item) for item in response.json()]
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP error fetching filtered projects from data-service: {e.response.status_code} - {e.response.text}")
            return []
        except httpx.RequestError as e:
            print(f"❌ Network error fetching filtered projects from data-service: {e}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred while fetching filtered projects: {e}")
            return []

    async def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Fetches a single project's details by ID from the data-service."""
        try:
            response = await self.client.get(f"/projects/{project_id}", timeout=10)
            response.raise_for_status()
            return Project(**response.json())
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f"Project with ID '{project_id}' not found in data-service.")
                return None
            print(f"❌ HTTP error fetching project by ID from data-service: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            print(f"❌ Network error fetching project by ID from data-service: {e}")
            return None
        except Exception as e:
            print(f"❌ An unexpected error occurred while fetching project by ID: {e}")
            return None

    async def get_all_project_ids(self) -> List[str]:
        """Fetches all project IDs from the data-service."""
        try:
            response = await self.client.get("/projects/all_ids", timeout=10)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"❌ Network error fetching all project IDs from data-service: {e}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred while fetching all project IDs: {e}")
            return []