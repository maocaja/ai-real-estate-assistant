import openai
from typing import List, Dict, Any, Optional

from chat_api.config.settings import settings
from chat_api.models.chat_models import Message, LLMRequest, LLMResponse # Reusamos los modelos de chat

class LLMClient:
    def __init__(self):
        # Configura la API key de OpenAI. Asegúrate de que settings.OPENAI_API_KEY tenga tu clave real.
        openai.api_key = settings.OPENAI_API_KEY
        self.model_name = settings.OPENAI_MODEL_NAME
        
        # Validar que la API key no sea la predeterminada si estamos en un entorno que no sea de desarrollo
        if settings.OPENAI_API_KEY == "your_openai_api_key_here" and os.getenv("APP_ENV") != "development":
            print("⚠️ WARNING: OpenAI API key is not set. Using default placeholder. This will likely cause authentication errors.")
            # En un entorno de producción, aquí podrías levantar una excepción o detener el servicio.

    async def get_chat_completion(self, messages: List[Message], temperature: float = 0.7) -> LLMResponse:
        """
        Hace una llamada a la API de OpenAI para obtener una respuesta del chat.
        """
        if not openai.api_key or openai.api_key == "your_openai_api_key_here":
            raise ValueError("OpenAI API key is not set. Cannot call LLM.")

        try:
            # Convierte la lista de modelos Message a un formato que la API de OpenAI entiende
            # (que son diccionarios con 'role' y 'content')
            formatted_messages = [msg.model_dump() for msg in messages]

            # Llama a la API de OpenAI de forma asíncrona
            # Nota: La librería 'openai' maneja las llamadas asíncronas con 'await client.chat.completions.create'
            # para versiones más recientes (>=1.0). Para versiones antiguas (<1.0), es 'await openai.ChatCompletion.acreate'
            # Asegúrate de que tu versión de 'openai' sea compatible.
            
            # Usando la nueva sintaxis (openai >= 1.0)
            client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=temperature,
            )
            
            # Extrae el contenido de la respuesta
            response_content = response.choices[0].message.content
            return LLMResponse(response_content=response_content)

        except openai.AuthenticationError as e:
            print(f"❌ OpenAI API Authentication Error: {e}. Check your API key.")
            raise ValueError("Authentication with LLM failed. Please check API key.") from e
        except openai.APIError as e:
            print(f"❌ OpenAI API Error: {e}")
            raise RuntimeError(f"Error calling LLM API: {e}") from e
        except Exception as e:
            print(f"❌ An unexpected error occurred while calling LLM: {e}")
            raise RuntimeError(f"Unexpected error with LLM: {e}") from e