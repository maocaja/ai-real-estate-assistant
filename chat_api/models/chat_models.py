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


class ProjectUnit(BaseModel):
    name: Optional[str] = Field(None, alias="nombre")
    bedrooms: Optional[int] = Field(None, alias="habitaciones")
    bathrooms: Optional[int] = Field(None, alias="baños")
    area_sqm: Optional[float] = Field(None, alias="area_m2")
    price_from: Optional[float] = Field(None, alias="precio")
    parking: Optional[bool] = Field(None, alias="parqueadero")
    balcony: Optional[bool] = Field(None, alias="balcon")
    floor_material: Optional[str] = Field(None, alias="piso")
    estimated_annual_rent_income: Optional[float] = Field(None, alias="ingreso_anual_arriendo_estimado")
    estimated_annual_expenses: Optional[float] = Field(None, alias="gastos_anuales_estimados")
    estimated_annual_appreciation_rate: Optional[float] = Field(None, alias="tasa_valorizacion_anual_estimada")
    estimated_investment_horizon_years: Optional[int] = Field(None, alias="horizonte_inversion_anos_estimado")

class Coordinates(BaseModel):
    lat: float
    lng: float

class Project(BaseModel):
    id: str
    # Campos que son requeridos y tienen alias
    project_name: str = Field(..., alias="nombre_proyecto")
    builder: str = Field(..., alias="constructor")
    status: str = Field(..., alias="estado")
    city: str = Field(..., alias="ciudad")
    zone: str = Field(..., alias="zona")
    general_description: str = Field(..., alias="descripcion_general")
    units: List[ProjectUnit] = Field(..., alias="unidades") # Asumiendo que las unidades siempre están presentes y son una lista

    # Campos que son opcionales y tienen alias
    amenities: Optional[List[str]] = Field(None, alias="amenidades") # Corregido: Optional y Field(None, ...)
    estimated_delivery_date: Optional[str] = Field(None, alias="fecha_entrega_estimado")
    coordinates: Optional[Coordinates] = Field(None, alias="coordenadas")
    image_url: Optional[str] = Field(None, alias="imagen_url") # Mantengo str como en tu modelo
    socioeconomic_level: Optional[float] = Field(None, alias="estrato")
    short_term_rental: Optional[bool] = Field(None, alias="renta_corta")
    type_property: Optional[str] = Field(None, alias="tipo") # Mantengo Optional como en tu modelo
    area_min_square: Optional[float] = Field(None, alias="area_minima_m2_desde")
    bathroom_min: Optional[float] = Field(None, alias="banios_minimo_desde")
    bedrooms_min: Optional[float] = Field(None, alias="habitaciones_minimo_desde")
    price_min: Optional[float] = Field(None, alias="precio_minimo_desde")
    price_max: Optional[float] = Field(None, alias="precio_maximo_hasta")
    recommended_use: Optional[str] = Field(None, alias="tipo_uso_recomendado")

    class Config:
        populate_by_name = True # Permite usar tanto el nombre del campo como el alias
        # Puedes añadir `extra = "ignore"` si quieres ignorar campos en el input que no estén definidos en el modelo,
        # o `extra = "forbid"` si quieres que falle si hay campos no definidos.
        # Por defecto, Pydantic v2 es bastante estricto.


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