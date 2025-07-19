from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from dotenv import load_dotenv
load_dotenv()

from chat_api.api.routes import router # Importa el router de las rutas
from chat_api.config.settings import settings
from chat_api.services.llm_client import LLMClient # Importa LLMClient para la comprobación inicial de la clave API

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for application startup and shutdown events.
    """
    print("🚀 Chat API starting up...")
    
    # Optional: Initial check for LLM API key at startup
    if settings.OPENAI_API_KEY == "your_openai_api_key_here" or not settings.OPENAI_API_KEY:
        print("🚨 WARNING: OpenAI API key is not configured. LLM calls will fail.")
        # En un entorno de producción, podrías optar por levantar una excepción aquí
        # para evitar que el servicio se inicie sin la configuración crítica.

    # Optional: Initial health checks for dependent services
    # No es estrictamente necesario aquí si el /health endpoint ya los comprueba,
    # pero puede ser útil para logs de startup.
    # try:
    #     print("Checking Data Service availability...")
    #     # await chat_service_instance.data_service_client.get_all_project_ids() # A simple call
    #     print("Data Service seems available.")
    # except Exception as e:
    #     print(f"❌ Data Service might be unavailable: {e}")

    # try:
    #     print("Checking Embedding Worker availability...")
    #     health_status = await chat_service_instance.embedding_service_client.get_health()
    #     if health_status.get("status") != "OK":
    #         print(f"❌ Embedding Worker is not fully ready: {health_status.get('error', health_status.get('status'))}")
    #     else:
    #         print("Embedding Worker seems available and ready.")
    # except Exception as e:
    #     print(f"❌ Embedding Worker might be unavailable: {e}")


    yield # Application is running

    # Shutdown events (none specific needed for chat_api for now, but placeholder)
    print("🛬 Chat API shutting down...")

app = FastAPI(
    title="Chat API - AI Real Estate Assistant",
    description="Microservice for handling user chat interactions and orchestrating AI components.",
    version="1.0.0",
    lifespan=lifespan # Link the lifespan function here
)

# Incluye las rutas de la API
app.include_router(router)

@app.get("/")
async def read_root():
    """Simple root endpoint for health check."""
    return {"message": "Chat API is running!"}