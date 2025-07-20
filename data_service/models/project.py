from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Define a Pydantic model for a single unit within a project
class ProjectUnit(BaseModel):
    name: Optional[str] = Field(None, alias="nombre")
    bedrooms: Optional[int] = Field(None, alias="habitaciones")
    bathrooms: Optional[int] = Field(None, alias="baños")
    area_sqm: Optional[float] = Field(None, alias="area_m2")
    price_from: Optional[float] = Field(None, alias="precio")
    parking: Optional[bool] = Field(None, alias="parqueadero")
    balcony: Optional[bool] = Field(None, alias="balcon")
    floor_material: Optional[str] = Field(None, alias="piso")
    # Added fields from enriched data (if applicable)
    estimated_annual_rent_income: Optional[float] = Field(None, alias="ingreso_anual_arriendo_estimado")
    estimated_annual_expenses: Optional[float] = Field(None, alias="gastos_anuales_estimados")
    estimated_annual_appreciation_rate: Optional[float] = Field(None, alias="tasa_valorizacion_anual_estimada")
    estimated_investment_horizon_years: Optional[int] = Field(None, alias="horizonte_inversion_anos_estimado")

class Coordinates(BaseModel):
    lat: float
    lng: float

# Define a Pydantic model for a full project
class Project(BaseModel):
    id: str
    project_name: str = Field(..., alias="nombre_proyecto")
    builder: str = Field(..., alias="constructor")
    status: str = Field(..., alias="estado")
    city: str = Field(..., alias="ciudad")
    zone: str = Field(..., alias="zona")
    general_description: str = Field(..., alias="descripcion_general")
    units: List[ProjectUnit] = Field(..., alias="unidades") # List of units within the project
    amenities: Optional[List[str]] = Field(..., alias="amenidades")
    estimated_delivery_date: Optional[str] = Field(None, alias="fecha_entrega_estimado")
    coordinates: Optional[Coordinates] = Field(None, alias="coordenadas")
    image_url: Optional[str] = Field(None, alias="imagen_url")
    socioeconomic_level: Optional[float] = Field(None, alias="estrato")
    short_term_rental: Optional[bool] = Field(None, alias="renta_corta")
    type_property: Optional[str] = Field(None, alias="tipo")
    area_min_square: Optional[float] = Field(None, alias="area_minima_m2_desde")
    bathroom_min: Optional[float] = Field(None, alias="banios_minimo_desde")
    bedrooms_min: Optional[float] = Field(None, alias="habitaciones_minimo_desde")
    price_min: Optional[float] = Field(None, alias="precio_minimo_desde")
    price_max: Optional[float] = Field(None, alias="precio_maximo_hasta")
    recommended_use: Optional[str] = Field(None, alias="tipo_uso_recomendado")

# Model for amenities data specifically for embedding worker
class ProjectAmenitiesData(BaseModel):
    id: str
    amenidades_text: str # Combined amenities as a single string
