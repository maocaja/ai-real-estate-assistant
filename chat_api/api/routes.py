from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional

from chat_api.models.chat_models import ChatRequest, ChatResponse, Message
from chat_api.services.chat_service import ChatService
from chat_api.services.data_client import DataServiceClient
from chat_api.services.embedding_client import EmbeddingServiceClient
from chat_api.services.llm_client import LLMClient

router = APIRouter()

# Instanciar el ChatService. Esto es un buen lugar para pasarlo como dependencia
# o instanciarlo globalmente si no tiene estado por request.
# Para simplicidad inicial, lo instanciamos aquí.
# Una mejor práctica para FastAPI es usar Dependencias para los servicios,
# lo que permite inyección de dependencias y testing más fácil.
# Por ahora, para simplificar el arranque, lo instanciamos directamente.

# Instancia global del ChatService
# Esto asegura que los clientes internos (LLM, Data, Embedding) se inicialicen una vez.
chat_service_instance = ChatService()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    Handles a chat message from the user and returns a response from the AI assistant.
    """
    try:
        response = await chat_service_instance.handle_message(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad request: {e}")
    except Exception as e:
        print(f"❌ Unhandled error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.get("/health")
async def health_check():
    """
    Provides a health check for the chat API and its dependent services.
    """
    data_service_status = "OK"
    embedding_worker_status = "OK"
    llm_service_status = "OK" # Assumed OK unless an explicit check is available for LLM API status

    # Check Data Service health (optional, but good for robust health checks)
    try:
        # We don't have a direct health endpoint for data-service in data_service/api/routes.py
        # A simple request to a base endpoint or /projects/all_ids can serve as a proxy.
        # For simplicity, we assume if client initializes, it should be able to connect.
        # A better health check would be to add a /health endpoint to data-service.
        pass # For now, no direct health check for data_service_client
    except Exception as e:
        data_service_status = f"Degraded ({e})"

    # Check Embedding Worker health
    embedding_health_info = await chat_service_instance.embedding_service_client.get_health()
    if embedding_health_info.get("status") != "OK":
        embedding_worker_status = f"Degraded ({embedding_health_info.get('error', 'Unknown Embedding Worker issue')})"
    elif not embedding_health_info.get("faiss_index_built"):
        embedding_worker_status = "Degraded (Embedding index not built)"


    return {
        "status": "OK",
        "message": "Chat API is running!",
        "dependent_services": {
            "data_service": data_service_status,
            "embedding_worker": embedding_worker_status,
            "llm_client": llm_service_status # Could add more checks if LLM client had a direct health method
        }
    }