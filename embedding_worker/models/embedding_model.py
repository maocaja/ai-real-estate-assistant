from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SearchRequest(BaseModel):
    query_text: str
    project_ids_subset: Optional[List[str]] = None # Para búsquedas dentro de un subconjunto
    limit: int = 5 # Añade este campo para controlar el número de resultados

class SearchResponse(BaseModel):
    similar_project_ids: List[str]

class IndexDataRequest(BaseModel):
    """
    Represents the structure of data fetched from data-service for indexing.
    This is an internal model to define the expected format from data-service's amenities_data endpoint.
    """
    id: str
    amenidades_text: str # Combined amenities as a single string

