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
        """Loads project data from the configured JSON path."""
        try:
            df = pd.read_json(settings.DATA_PATH, orient="records")
            df['id'] = df['id'].astype(str) # Ensure 'id' column is string
            self._projects_df = df
            self._projects_dict = df.set_index('id').to_dict(orient='index')
            print(f"✅ Data loaded from {settings.DATA_PATH}. Total projects: {len(self._projects_df)}")
        except FileNotFoundError:
            print(f"❌ Error: Data file not found at {settings.DATA_PATH}.")
            raise # Re-raise to signal critical startup failure
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise # Re-raise for any other loading errors

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
        if city:
            filtered_df = filtered_df[filtered_df['ciudad'].str.contains(city, case=False, na=False)]
        if property_type:
            filtered_df = filtered_df[filtered_df['tipo_propiedad'].str.contains(property_type, case=False, na=False)]
        if min_price is not None:
            filtered_df = filtered_df[filtered_df['precio_desde'] >= min_price] # Assuming 'precio_desde' is the price field
        if max_price is not None:
            filtered_df = filtered_df[filtered_df['precio_desde'] <= max_price] # Assuming 'precio_desde' is the price field
        if min_bedrooms is not None:
            filtered_df = filtered_df[filtered_df['habitaciones'] >= min_bedrooms]
        if min_bathrooms is not None:
            filtered_df = filtered_df[filtered_df['baños'] >= min_bathrooms]
        if min_area_sqm is not None:
            filtered_df = filtered_df[filtered_df['area_m2'] >= min_area_sqm]
        if recommended_use:
            filtered_df = filtered_df[filtered_df['tipo_uso_recomendado'].str.contains(recommended_use, case=False, na=False)]
        if project_name:
            filtered_df = filtered_df[filtered_df['nombre_proyecto'].str.contains(project_name, case=False, na=False)]

        return filtered_df.head(settings.PROJECT_RESULTS_LIMIT).to_dict(orient='records')

    def get_project_by_id(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single project by its unique ID."""
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