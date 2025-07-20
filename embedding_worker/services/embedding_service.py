# embedding_worker/services/embedding_service.py

import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer
import faiss # Importar FAISS
import httpx # Para comunicarse con el data-service
from pydantic import BaseModel, Field

from data_service.models.project import Project # Para validar los datos del proyecto
from embedding_worker.config.settings import settings

# Clase Singleton para almacenar el modelo y el índice FAISS
class EmbeddingService:
    _instance = None
    _model: Optional[SentenceTransformer] = None
    _faiss_index: Optional[faiss.IndexFlatL2] = None
    _project_id_map: Dict[int, str] = {} # Mapea el índice FAISS al ID del proyecto
    _is_index_built: bool = False # Bandera para indicar si el índice está listo

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            # No cargar el modelo/índice aquí, sino en un método asíncrono
            # para que FastAPI pueda gestionar el ciclo de vida.
        return cls._instance

    async def initialize(self):
        """
        Inicializa el modelo de embeddings y construye el índice FAISS.
        Debe ser llamado durante el startup de la aplicación.
        """
        if self._model is None:
            print(f"⏳ Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            try:
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                print("✅ Embedding model loaded.")
            except Exception as e:
                print(f"❌ Error loading embedding model: {e}")
                self._model = None
                return

        if not self._is_index_built:
            print("⏳ Building FAISS index for projects...")
            await self._build_faiss_index()
            if self._is_index_built:
                print("✅ FAISS index built successfully.")
            else:
                print("❌ Failed to build FAISS index.")

    def get_model(self) -> Optional[SentenceTransformer]:
        return self._model

    def is_index_ready(self) -> bool:
        return self._is_index_built

    async def _get_all_projects_data(self) -> List[Project]:
        """
        Obtiene todos los proyectos del data-service.
        """
        async with httpx.AsyncClient(base_url=settings.DATA_SERVICE_URL) as client:
            try:
                response = await client.get("/projects", timeout=60) # Increased timeout
                response.raise_for_status()
                # Validar cada item en la lista usando Project.model_validate
                return [Project.model_validate(item) for item in response.json()]
            except httpx.HTTPStatusError as e:
                print(f"❌ HTTP error fetching all projects from data-service: {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:
                print(f"❌ Network error fetching all projects from data-service: {e}")
            except Exception as e:
                print(f"❌ An unexpected error occurred while fetching all projects for embedding: {e}")
        return []

    async def _build_faiss_index(self):
        """
        Construye el índice FAISS con embeddings de todos los proyectos.
        """
        if self._model is None:
            print("❌ Embedding model not loaded, cannot build FAISS index.")
            return

        projects = await self._get_all_projects_data()
        if not projects:
            print("❌ No projects loaded from data-service to build FAISS index.")
            return

        texts_to_embed = []
        project_ids = []

        for project in projects:
            # Construir una cadena de texto representativa para cada proyecto
            # Usa los nombres de los atributos de Pydantic del modelo Project
            description_parts = [
                project.general_description if project.general_description else "",
                project.city if project.city else "",
                project.zone if project.zone else "",
                project.type_property if project.type_property else "",
                f"Precios desde ${project.price_min}" if project.price_min else "",
                f"hasta ${project.price_max}" if project.price_max else "",
                f"{int(project.bedrooms_min)} habitaciones" if project.bedrooms_min else "",
                f"{int(project.bathroom_min)} baños" if hasattr(project, 'bathroom_min') and project.bathroom_min else "",
                f"{int(project.area_min_square)} m2" if hasattr(project, 'area_min_square') and project.area_min_square else "",
                f"Uso: {project.recommended_use}" if project.recommended_use else ""
            ]
            if project.amenities:
                if isinstance(project.amenities, list):
                    description_parts.append("Amenidades: " + ", ".join(project.amenities))
                else:
                    description_parts.append(f"Amenidades: {project.amenities}")
            
            # Filtra partes vacías y únelas
            full_description = ". ".join(filter(None, description_parts)).strip()
            
            if full_description: # Solo si hay una descripción significativa
                texts_to_embed.append(full_description)
                project_ids.append(project.id)
            else:
                print(f"Skipping project {project.id} due to empty description for embedding.")


        if not texts_to_embed:
            print("❌ No valid project descriptions to embed for FAISS index.")
            return

        print(f"Generating embeddings for {len(texts_to_embed)} projects...")
        embeddings = self._model.encode(texts_to_embed, show_progress_bar=True)
        
        # FAISS requiere numpy arrays de tipo float32
        embeddings = np.array(embeddings).astype('float32')
        
        # Crear un índice FAISS FlatL2 (L2 distance, Euclidean distance)
        # d es la dimensión del embedding (ej. 768 para all-MiniLM-L6-v2)
        d = embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatL2(d)
        self._faiss_index.add(embeddings)

        self._project_id_map = {i: proj_id for i, proj_id in enumerate(project_ids)}
        self._is_index_built = True
        print(f"✅ FAISS index built with {self._faiss_index.ntotal} vectors.")


    async def search_similar_projects(self, query_text: str, limit: int = 5) -> List[str]:
        """
        Busca proyectos similares en el índice FAISS.
        """
        if not self._is_index_built or self._model is None:
            print("❌ FAISS index not built or model not loaded, cannot perform search.")
            return []

        # Generar embedding para la consulta
        query_embedding = self._model.encode([query_text]).astype('float32')
        
        # Realizar la búsqueda en FAISS
        # D: distancias, I: índices de los vecinos más cercanos
        D, I = self._faiss_index.search(query_embedding, limit)
        
        # Mapear los índices FAISS a los IDs de proyecto originales
        similar_project_ids = []
        for faiss_index in I[0]: # I[0] porque search devuelve un array de arrays (uno por cada query)
            if faiss_index != -1: # -1 indica que no se encontró un vecino (puede pasar si limit > ntotal)
                project_id = self._project_id_map.get(faiss_index)
                if project_id:
                    similar_project_ids.append(project_id)
        
        return similar_project_ids