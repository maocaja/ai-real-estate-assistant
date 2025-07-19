from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional

from data_service.repository.project_repository import ProjectRepository
from data_service.models.project import Project, ProjectAmenitiesData # Import models

router = APIRouter()
project_repo = ProjectRepository() # Initialize the repository

@router.get("/projects", response_model=List[Project]) # Use Project model for response
async def get_projects(
    city: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_bedrooms: Optional[int] = None,
    min_bathrooms: Optional[int] = None,
    min_area_sqm: Optional[float] = None,
    recommended_use: Optional[str] = None,
    project_name: Optional[str] = None
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
        project_name=project_name
    )
    return projects


@router.get("/projects/{project_id}", response_model=Project) # Use Project model for response
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


@router.get("/projects/all_ids", response_model=List[str])
async def get_all_project_identifiers():
    """
    Returns a list of all available project IDs.
    """
    return project_repo.get_all_project_ids()


@router.get("/projects/amenities_data", response_model=List[ProjectAmenitiesData]) # Use ProjectAmenitiesData model
async def get_project_amenities_for_embedding():
    """
    Returns project IDs and their amenities text,
    formatted for the embedding worker.
    """
    return project_repo.get_project_amenities_data()