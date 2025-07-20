import pandas as pd
from typing import List, Dict, Any, Optional
from data_service.config.settings import settings # Import settings

class ProjectRepository:
    """
    Manages loading and querying project data from the configured source.
    Implements the Singleton pattern.
    """
    _instance = None # Singleton instance
    _projects_df: pd.DataFrame = pd.DataFrame()
    _projects_dict: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        """Ensures a single instance of the repository."""
        if cls._instance is None:
            cls._instance = super(ProjectRepository, cls).__new__(cls)
            cls._instance._load_data() # Load data upon first instantiation
        return cls._instance

    def _load_data(self):
        try:
            df = pd.read_json(settings.DATA_PATH, orient="records")
            df['id'] = df['id'].astype(str)
            self._projects_df = df # Correct assignment
            self._projects_dict = df.set_index('id').to_dict(orient='index')
            print(f"✅ Data loaded from {settings.DATA_PATH}. Total projects: {len(self._projects_df)}")

            # --- COMIENZO DE LAS NUEVAS LÍNEAS DE DEPURACIÓN - CORREGIDAS ---
            print("\nRevisando tipos de datos y recuentos de valores no nulos en columnas clave:")
            for col in ['precio_minimo_desde', 'precio_maximo_hasta', 'habitaciones_minimo_desde', 'banios_minimo_desde', 'area_minima_m2_desde', 'tipo', 'ciudad', 'nombre_proyecto', 'tipo_uso_recomendado']:
                # Change self._df to self._projects_df here and below
                if col in self._projects_df.columns: # CORRECTED
                    print(f"  - Columna '{col}': Tipo de dato (dtype): {self._projects_df[col].dtype}, No nulos: {self._projects_df[col].count()}/{len(self._projects_df)}") # CORRECTED
                    if 'precio' in col or 'habitaciones' in col or 'banios' in col or 'area' in col:
                        numeric_series = pd.to_numeric(self._projects_df[col], errors='coerce') # CORRECTED
                        if numeric_series.isnull().any() and self._projects_df[col].count() > numeric_series.count(): # CORRECTED
                            print(f"    ¡ADVERTENCIA! La columna '{col}' contiene valores no numéricos que fueron convertidos a NaN.")
                            print(f"    Ejemplos de valores no numéricos: {self._projects_df[col][numeric_series.isnull()].unique()[:5]}") # CORRECTED
                else:
                    print(f"  - ¡ADVERTENCIA! La columna '{col}' NO SE ENCONTRÓ en el DataFrame.")
            print("-" * 50)
            # --- FIN DE LAS NUEVAS LÍNEAS DE DEPURACIÓN ---
        except FileNotFoundError:
            print(f"❌ Error: Data file not found at {settings.DATA_PATH}.")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise

    def is_data_loaded(self) -> bool:
        """Checks if the project data has been successfully loaded."""
        return not self._projects_df.empty

    def get_filtered_projects(
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
    ) -> List[Dict[str, Any]]:
        """Applies filters to the project DataFrame and returns a limited list of results."""
        filtered_df = self._projects_df.copy()

        # Apply filters conditionally (using original column names from JSON)
        print("\n--- Aplicando Filtros ---") # New debug line
        print(f"Número inicial de proyectos: {len(filtered_df)}") # New debug line

        if city:
            print(f"Aplicando filtro por Ciudad: '{city}'") # New debug line
            filtered_df = filtered_df[filtered_df['ciudad'].str.contains(city, case=False, na=False)]
            print(f"Proyectos restantes después de Ciudad: {len(filtered_df)}") # New debug line
        if property_type:
            print(f"Aplicando filtro por Tipo de Propiedad: '{property_type}'") # New debug line
            filtered_df = filtered_df[filtered_df['tipo'].str.contains(property_type, case=False, na=False)]
            print(f"Proyectos restantes después de Tipo de Propiedad: {len(filtered_df)}") # New debug line
        if min_price is not None:
            print(f"Aplicando filtro por Precio Mínimo: '{min_price}'") # New debug line
            filtered_df = filtered_df[filtered_df['precio_minimo_desde'] >= min_price]
            print(f"Proyectos restantes después de Precio Mínimo: {len(filtered_df)}") # New debug line
        if max_price is not None:
            print(f"Aplicando filtro por Precio Máximo: '{max_price}' (usando precio_minimo_desde)") # New debug line
            # This is the line we've been working on, ensure it's changed to precio_minimo_desde
            filtered_df = filtered_df[filtered_df['precio_minimo_desde'] <= max_price]
            print(f"Proyectos restantes después de Precio Máximo: {len(filtered_df)}") # New debug line
        if min_bedrooms is not None:
            print(f"Aplicando filtro por Habitaciones Mínimas: '{min_bedrooms}'") # New debug line
            filtered_df = filtered_df[filtered_df['habitaciones_minimo_desde'] >= min_bedrooms]
            print(f"Proyectos restantes después de Habitaciones Mínimas: {len(filtered_df)}") # New debug line
        if min_bathrooms is not None:
            print(f"Aplicando filtro por Baños Mínimos: '{min_bathrooms}'") # New debug line
            filtered_df = filtered_df[filtered_df['baños'] >= min_bathrooms]
            print(f"Proyectos restantes después de Baños Mínimos: {len(filtered_df)}") # New debug line
        if min_area_sqm is not None:
            print(f"Aplicando filtro por Área Mínima (m²): '{min_area_sqm}'") # New debug line
            filtered_df = filtered_df[filtered_df['area_m2'] >= min_area_sqm]
            print(f"Proyectos restantes después de Área Mínima: {len(filtered_df)}") # New debug line
        if recommended_use:
            print(f"Aplicando filtro por Uso Recomendado: '{recommended_use}'") # New debug line
            filtered_df = filtered_df[filtered_df['tipo_uso_recomendado'].str.contains(recommended_use, case=False, na=False)]
            print(f"Proyectos restantes después de Uso Recomendado: {len(filtered_df)}") # New debug line
        if project_name:
            print(f"Aplicando filtro por Nombre de Proyecto: '{project_name}'") # New debug line
            filtered_df = filtered_df[filtered_df['nombre_proyecto'].str.contains(project_name, case=False, na=False)]
            print(f"Proyectos restantes después de Nombre de Proyecto: {len(filtered_df)}") # New debug line

        print("--- Fin de Aplicación de Filtros ---\n") # New debug line

        return filtered_df.head(settings.PROJECT_RESULTS_LIMIT).to_dict(orient='records')

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single project by its unique ID."""
        print(f"Returning from repository: {self._projects_dict.get(project_id).keys() if self._projects_dict.get(project_id) else 'None'}")
        return self._projects_dict.get(project_id)

    def get_all_project_ids(self) -> List[str]:
        """Returns a list of all available project IDs."""
        return self._projects_df['id'].tolist() if self.is_data_loaded() else []

    def get_project_amenities_data(self) -> List[Dict[str, Any]]:
        """
        Returns project IDs and their amenities text,
        formatted for the embedding worker.
        """
        if not self.is_data_loaded() or 'amenidades' not in self._projects_df.columns:
            if not self.is_data_loaded():
                print("Warning: Data not loaded, cannot provide amenities data.")
            else:
                print("Warning: 'amenidades' column not found in the dataset.")
            return []

        amenities_data = []
        for _, row in self._projects_df[['id', 'amenidades']].iterrows():
            amenity_value = row['amenidades']
            if isinstance(amenity_value, list):
                amenity_value = ", ".join(amenity_value)
            elif not isinstance(amenity_value, str):
                amenity_value = str(amenity_value)
            amenities_data.append({
                "id": row['id'],
                "amenidades_text": amenity_value
            })
        return amenities_data