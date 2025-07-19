import openai
from typing import List, Dict, Any, Optional
import os

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

    async def get_chat_completion(self, messages: List[Message], tools: Optional[List[Dict[str, Any]]] = None, temperature: float = 0.7) -> LLMResponse:
        """
        Hace una llamada a la API de OpenAI para obtener una respuesta del chat,
        con soporte para tool_calling.
        """
        if not openai.api_key or openai.api_key == "your_openai_api_key_here":
            raise ValueError("OpenAI API key is not set. Cannot call LLM.")

        try:
            formatted_messages = [msg.model_dump() for msg in messages]

            client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            # Prepara los argumentos para la llamada a la API de OpenAI
            api_call_kwargs = {
                "model": self.model_name,
                "messages": formatted_messages,
                "temperature": temperature,
            }

            if tools:
                api_call_kwargs["tools"] = tools
                # Opcional: force tool calling if you always want it to try to use a tool
                # api_call_kwargs["tool_choice"] = "auto" # Default, LLM decides
                # api_call_kwargs["tool_choice"] = {"type": "function", "function": {"name": "your_function_name"}} # Force a specific tool

            response = await client.chat.completions.create(**api_call_kwargs)

            # Procesar la respuesta para ver si es texto o una llamada a función
            choice = response.choices[0]
            if choice.message.content:
                return LLMResponse(response_content=choice.message.content)
            elif choice.message.tool_calls:
                # Convierte los objetos ToolCall a diccionarios planos para nuestro modelo
                tool_calls_dicts = [tc.model_dump() for tc in choice.message.tool_calls]
                return LLMResponse(tool_calls=tool_calls_dicts)
            else:
                return LLMResponse(response_content="Lo siento, no pude generar una respuesta o una llamada a herramienta.")

        except openai.AuthenticationError as e:
            print(f"❌ OpenAI API Authentication Error: {e}. Check your API key.")
            raise ValueError("Authentication with LLM failed. Please check API key.") from e
        except openai.APIError as e:
            print(f"❌ OpenAI API Error: {e}")
            raise RuntimeError(f"Error calling LLM API: {e}") from e
        except Exception as e:
            print(f"❌ An unexpected error occurred while calling LLM: {e}")
            raise RuntimeError(f"Unexpected error with LLM: {e}") from e