import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any, Optional

from data_service.models.project import ProjectAmenitiesData # Import the model from data-service
from embedding_worker.config.settings import settings

class EmbeddingRepository:
    """
    Manages embedding model loading, FAISS index creation,
    and similarity search operations.
    Implements the Singleton pattern.
    """
    _instance = None
    _model: Optional[SentenceTransformer] = None
    _faiss_index: Optional[faiss.Index] = None
    _project_id_map: List[str] = [] # Maps FAISS index IDs back to original project IDs

    def __new__(cls):
        """Ensures a single instance of the repository."""
        if cls._instance is None:
            cls._instance = super(EmbeddingRepository, cls).__new__(cls)
            cls._instance._load_embedding_model()
        return cls._instance

    def _load_embedding_model(self):
        """Loads the pre-trained Sentence Transformer model."""
        try:
            # Load model with local cache directory
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, cache_folder=settings.MODEL_CACHE_DIR)
            print(f"✅ Embedding model '{settings.EMBEDDING_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise # Critical error, should halt startup

    def is_model_loaded(self) -> bool:
        """Checks if the embedding model is loaded."""
        return self._model is not None

    def is_index_built(self) -> bool:
        """Checks if the FAISS index is built and not empty."""
        return self._faiss_index is not None and self._faiss_index.ntotal > 0

    def fetch_data_for_indexing(self) -> List[Dict[str, Any]]:
        """
        Fetches project amenities data from the data-service for building the FAISS index.
        """
        try:
            response = requests.get(f"{settings.DATA_SERVICE_URL}/projects/amenities_data", timeout=30)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            raw_data = response.json()

            # Validate and process data using Pydantic model
            processed_data = [ProjectAmenitiesData(**item).model_dump() for item in raw_data]
            print(f"Fetched {len(processed_data)} items for indexing from {settings.DATA_SERVICE_URL}.")
            return processed_data
        except requests.exceptions.ConnectionError as e:
            print(f"❌ Connection error fetching data from data-service: {e}. Is data-service running at {settings.DATA_SERVICE_URL}?")
            raise Exception("Could not connect to data-service.") # Re-raise as generic exception for repo
        except requests.exceptions.Timeout:
            print("❌ Timeout error fetching data from data-service.")
            raise Exception("Timeout while fetching data from data-service.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data from data-service: {e}")
            raise Exception(f"Error fetching data from data-service: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred during data fetching: {e}")
            raise Exception(f"An unexpected error occurred: {e}")

    def build_faiss_index(self, projects_for_indexing: List[Dict[str, Any]]):
        """
        Builds or rebuilds the FAISS index from the provided project data.
        Each item in projects_for_indexing must have 'id' and 'amenidades_text'.
        """
        if not self.is_model_loaded():
            raise Exception("Embedding model not loaded, cannot build FAISS index.")
        if not projects_for_indexing:
            print("No data provided to build FAISS index. Index will be empty.")
            self._faiss_index = None
            self._project_id_map = []
            return

        print(f"Building FAISS index for {len(projects_for_indexing)} projects...")

        texts_to_embed = [proj['amenidades_text'] for proj in projects_for_indexing]
        project_ids = [proj['id'] for proj in projects_for_indexing]

        # Generate embeddings
        embeddings = self._model.encode(texts_to_embed, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32') # Ensure float32 for FAISS

        d = embeddings.shape[1] # Get embedding dimension
        self._faiss_index = faiss.IndexFlatL2(d) # Use L2 distance for similarity
        self._faiss_index.add(embeddings)
        self._project_id_map = project_ids # Store the mapping of FAISS index to original IDs

        print(f"✅ FAISS index built successfully with dimension {d} and {self._faiss_index.ntotal} vectors.")

    def search_similar_projects(self, query_text: str, project_ids_subset: Optional[List[str]] = None, top_k: int = 10) -> List[str]:
        """
        Performs a similarity search using the FAISS index.
        Optionally filters results within a given subset of project IDs.
        """
        if not self.is_index_built():
            raise Exception("FAISS index not built or is empty.")
        if not self.is_model_loaded():
            raise Exception("Embedding model not loaded.")

        query_embedding = self._model.encode([query_text]).astype('float32')

        # Perform search on the entire index first.
        # We'll search for more items than requested in top_k to allow for filtering.
        k_search = min(max(top_k * 5, 50), self._faiss_index.ntotal) # Search for a larger pool

        D, I = self._faiss_index.search(query_embedding, k_search) # D = distances, I = indices

        similar_ids_raw = []
        for idx in I[0]: # I[0] contains indices for the first (and only) query
            if idx != -1: # -1 means no result found for that slot
                similar_ids_raw.append(self._project_id_map[idx])

        final_similar_ids = []
        if project_ids_subset:
            # Filter by the provided subset while maintaining similarity order
            for pid in similar_ids_raw:
                if pid in project_ids_subset:
                    final_similar_ids.append(pid)
                    if len(final_similar_ids) >= top_k: # Stop once we have enough
                        break
        else:
            final_similar_ids = similar_ids_raw[:top_k] # Just take top_k if no subset

        return final_similar_ids