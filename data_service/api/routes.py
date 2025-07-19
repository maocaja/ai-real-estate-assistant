from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional

from data_service.repository.project_repository import ProjectRepository
from data_service.models.project import Project, ProjectAmenitiesData # Import models

router = APIRouter()
project_repo = ProjectRepository() # Initialize the repository

# --- ESENCIAL: Este endpoint ESPECÍFICO debe ir ANTES del endpoint /{project_id} ---
@router.get("/projects/amenities_data", response_model=List[ProjectAmenitiesData]) # Use ProjectAmenitiesData model
async def get_project_amenities_for_embedding():
    """
    Returns project IDs and their amenities text,
    formatted for the embedding worker.
    """
    if not project_repo.is_data_loaded():
        # Consider returning 503 Service Unavailable if data is critical and not loaded
        # Or an empty list if it's acceptable for the client to handle no data
        raise HTTPException(status_code=503, detail="Data service not ready: Dataset not loaded.")
    return project_repo.get_project_amenities_data()

# --- Este endpoint GENERAL debe ir DESPUÉS de los endpoints más específicos con /projects/ ---
@router.get("/projects/{project_id}", response_model=Dict[str, Any])
async def get_project_details_by_id(project_id: str):
    """
    Retrieves full details for a specific project by its ID.
    """
    if not project_repo.is_data_loaded():
        raise HTTPException(status_code=503, detail="Data service not ready: Dataset not loaded.")

    project = project_repo.get_project_by_id(project_id)
    if project:
        return project
    raise HTTPException(status_code=404, detail=f"Project with ID '{project_id}' not found.")


@router.get("/projects", response_model=List[Dict[str, Any]])
async def get_projects(
    city: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_bedrooms: Optional[int] = None,
    min_bathrooms: Optional[int] = None,
    min_area_sqm: Optional[float] = None,
    recommended_use: Optional[str] = None,
    project_name: Optional[str] = None # Ensure this is here
):
    """
    Retrieves real estate projects based on structured filters.
    """
    if not project_repo.is_data_loaded():
        raise HTTPException(status_code=503, detail="Data service not ready: Dataset not loaded.")

    projects = project_repo.get_filtered_projects(
        city=city,
        property_type=property_type,
        min_price=min_price,
        max_price=max_price,
        min_bedrooms=min_bedrooms,
        min_bathrooms=min_bathrooms,
        min_area_sqm=min_area_sqm,
        recommended_use=recommended_use,
        project_name=project_name # Pass this to the repository
    )
    return projects


@router.get("/projects/all_ids", response_model=List[str])
async def get_all_project_identifiers():
    """
    Returns a list of all available project IDs.
    """
    if not project_repo.is_data_loaded():
        return [] # Return empty list if data not loaded
    return project_repo.get_all_project_ids()