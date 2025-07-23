from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Message(BaseModel):
    role: str
    content: Optional[str] = None  # <--- Here's the relevant part
    tool_calls: Optional[List[Dict[str, Any]]] = None # Assuming tool_calls are a list of dicts
    tool_call_id: Optional[str] = None # For tool responses
    name: Optional[str] = None # For tool responses
    
class ChatRequest(BaseModel):
    """
    Represents a request to the chat API.
    Contains the current user message and optional conversation history.
    """
    current_message: str
    conversation_history: List[Message] = [] # List of previous messages

class Project(BaseModel):
    id: str
    nombre_proyecto: str
    tipo: str
    ciudad: str
    precio_minimo_desde: Optional[float] = None
    precio_maximo_hasta: Optional[float] = None
    habitaciones_minimo_desde: Optional[int] = None
    habitaciones_maximo_hasta: Optional[int] = None
    banios_minimo_desde: Optional[int] = None
    banios_maximo_hasta: Optional[int] = None
    area_minima_m2_desde: Optional[float] = None
    area_maxima_m2_hasta: Optional[float] = None
    tipo_uso_recomendado: Optional[str] = None
    descripcion: Optional[str] = None
    url_proyecto: Optional[str] = None
    url_imagen: Optional[str] = None
    latitud: Optional[float] = None
    longitud: Optional[float] = None
    amenidades: Optional[List[str]] = []

class ChatResponse(BaseModel):
    response_message: str
    recommended_projects: List[Project] = []

class LLMRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: float = 0.7 # Default temperature for creative output

class LLMResponse(BaseModel):
    response_content: Optional[str] = None 
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Other potential fields from LLM response, e.g., token usage, finish reason