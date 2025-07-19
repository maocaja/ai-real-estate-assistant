from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SearchRequest(BaseModel):
    """
    Represents a request to perform a similarity search.
    """
    query_text: str
    # Optional list of project IDs to filter the search results within.
    # If empty, the search is performed across all indexed projects.
    project_ids_subset: List[str] = []

class SearchResponse(BaseModel):
    """
    Represents the response from a similarity search.
    """
    # List of project IDs found to be similar, ordered by similarity.
    similar_project_ids: List[str]

class IndexDataRequest(BaseModel):
    """
    Represents the structure of data fetched from data-service for indexing.
    This is an internal model to define the expected format from data-service's amenities_data endpoint.
    """
    id: str
    amenidades_text: str # Combined amenities as a single string