from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Define a Pydantic model for a single unit within a project
class ProjectUnit(BaseModel):
    name: Optional[str] = Field(None, alias="nombre")
    bedrooms: Optional[int] = Field(None, alias="habitaciones")
    bathrooms: Optional[int] = Field(None, alias="baños")
    area_sqm: Optional[float] = Field(None, alias="area_m2")
    price_from: Optional[float] = Field(None, alias="precio_desde")
    parking: Optional[bool] = Field(None, alias="parqueadero")
    balcony: Optional[bool] = Field(None, alias="balcon")
    floor_material: Optional[str] = Field(None, alias="piso")
    # Added fields from enriched data (if applicable)
    estimated_annual_rent_income: Optional[float] = Field(None, alias="ingreso_anual_arriendo_estimado")
    estimated_annual_expenses: Optional[float] = Field(None, alias="gastos_anuales_estimados")
    estimated_annual_appreciation_rate: Optional[float] = Field(None, alias="tasa_valorizacion_anual_estimada")
    estimated_investment_horizon_years: Optional[int] = Field(None, alias="horizonte_inversion_anos_estimado")

# Define a Pydantic model for a full project
class Project(BaseModel):
    id: str
    project_name: str = Field(..., alias="nombre_proyecto")
    builder: str = Field(..., alias="constructor")
    status: str = Field(..., alias="estado")
    city: str = Field(..., alias="ciudad")
    zone: str = Field(..., alias="zona")
    general_description: str = Field(..., alias="descripcion_general")
    amenities: Optional[List[str]] = None # This will be optional if not always present or not a list
    units: List[ProjectUnit] = Field(..., alias="unidades") # List of units within the project
    # Ensure other fields are included if they exist in your JSON and are relevant
    # For example, if 'precio' is at the top level, add:
    price: Optional[float] = Field(None, alias="precio")
    type_property: Optional[str] = Field(None, alias="tipo_propiedad")
    recommended_use: Optional[str] = Field(None, alias="tipo_uso_recomendado")


# Model for amenities data specifically for embedding worker
class ProjectAmenitiesData(BaseModel):
    id: str
    amenidades_text: str # Combined amenities as a single string