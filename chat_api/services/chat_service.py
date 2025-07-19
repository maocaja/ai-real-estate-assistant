from typing import List, Dict, Any, Optional

from chat_api.config.settings import settings
from chat_api.models.chat_models import Message, ChatRequest, ChatResponse, LLMResponse
from chat_api.services.llm_client import LLMClient
from chat_api.services.data_client import DataServiceClient
from chat_api.services.embedding_client import EmbeddingServiceClient

class ChatService:
    def __init__(self):
        self.llm_client = LLMClient()
        self.data_service_client = DataServiceClient()
        self.embedding_service_client = EmbeddingServiceClient()

        # Define el mensaje del sistema para guiar al LLM.
        # Esto le da al LLM su "rol" y algunas instrucciones iniciales.
        self.system_message_content = (
            "Eres un asistente de bienes raíces experto y amigable especializado en proyectos de vivienda "
            "en Colombia. Tu objetivo es ayudar a los usuarios a encontrar el proyecto ideal "
            "proporcionando información relevante, filtrando proyectos y respondiendo preguntas. "
            "Siempre responde de manera concisa y útil, ofreciendo solo la información solicitada. "
            "Si un usuario pide buscar un proyecto con ciertas características, primero busca proyectos "
            "usando las herramientas disponibles (filtrado de datos, búsqueda de similitud) y luego "
            "presenta la información relevante de esos proyectos."
            "No asumas información no proporcionada por las herramientas. Si no encuentras información "
            "relevante, dilo."
            "Recuerda que solo conoces proyectos de Colombia. Si te preguntan por otro país, indica que tu expertise es Colombia."
        )

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        """
        Maneja un mensaje entrante del usuario, orquestando las llamadas a los clientes
        y al LLM para generar una respuesta.
        """
        messages_for_llm: List[Message] = []

        # 1. Añadir el mensaje del sistema (guía para el LLM)
        messages_for_llm.append(Message(role="system", content=self.system_message_content))

        # 2. Añadir el historial de la conversación (para mantener el contexto)
        # Limita el historial para no exceder los límites de tokens del LLM y mantener la relevancia.
        # `settings.MAX_CONVERSATION_HISTORY` define cuántos turnos anteriores se incluyen.
        history_to_include = request.conversation_history[-settings.MAX_CONVERSATION_HISTORY:]
        messages_for_llm.extend(history_to_include)

        # 3. Añadir el mensaje actual del usuario
        messages_for_llm.append(Message(role="user", content=request.current_message))

        # --- LÓGICA DE ORQUESTACIÓN E INTENCIÓN (A EXPANDIR AQUÍ) ---
        # Esta es la parte clave donde integrarías los otros servicios.
        # Por ahora, es un ejemplo básico. En el futuro, aquí es donde:
        # 1. Se detectaría la intención del usuario (ej. ¿quiere filtrar por ciudad? ¿quiere buscar por amenidades?).
        # 2. Se extraerían las entidades (ej. "Bogotá", "apartamento", "gimnasio").
        # 3. Se decidiría qué cliente llamar (data-service para filtros, embedding-worker para búsqueda semántica).
        # 4. Se inyectaría la información recuperada en el contexto del LLM.

        # Ejemplo simplificado: Si el usuario menciona "Bogotá", intentamos buscar proyectos en Bogotá
        # ESTO ES UN EJEMPLO MUY BÁSICO, LA DETECCIÓN REAL NECESITA UN PARSER DE LENGUAJE NATURAL MÁS COMPLETO
        context_from_tools: Optional[str] = None
        relevant_project_ids: List[str] = []

        lower_message = request.current_message.lower()

        if "bogotá" in lower_message or "bogota" in lower_message:
            print("Detected 'Bogotá' in message. Attempting to filter projects.")
            # Llamar al data-service para filtrar por ciudad
            filtered_projects = await self.data_service_client.get_filtered_projects(city="Bogotá")
            if filtered_projects:
                project_names = [p.nombre_proyecto for p in filtered_projects[:3]] # Tomar solo los primeros 3 para el prompt
                context_from_tools = f"Se encontraron los siguientes proyectos en Bogotá: {', '.join(project_names)} y muchos más."
                relevant_project_ids = [p.id for p in filtered_projects]
            else:
                context_from_tools = "No se encontraron proyectos en Bogotá con las características actuales."

        elif "gimnasio" in lower_message or "piscina" in lower_message:
            print("Detected amenity keyword. Attempting semantic search.")
            # Llamar al embedding-worker para buscar proyectos similares
            similar_ids = await self.embedding_service_client.search_similar_projects(query_text=request.current_message, top_k=5)
            if similar_ids:
                # Opcional: Fetch full project details for these IDs if needed for rich LLM response
                # For simplicity here, just list the IDs
                context_from_tools = f"Se encontraron proyectos con amenidades similares a '{request.current_message}'. IDs relevantes: {', '.join(similar_ids)}."
                relevant_project_ids = similar_ids
            else:
                context_from_tools = "No se encontraron proyectos con esas amenidades."

        # Inyectar el contexto recuperado en el prompt del LLM si existe
        if context_from_tools:
            messages_for_llm.append(Message(role="system", content=f"INFORMACIÓN RELEVANTE DE HERRAMIENTAS:\n{context_from_tools}"))
            print(f"Added tool context to LLM: {context_from_tools}")


        # 4. Llamar al LLM para obtener la respuesta final
        try:
            llm_response: LLMResponse = await self.llm_client.get_chat_completion(
                messages=messages_for_llm,
                temperature=settings.OPENAI_LLM_TEMPERATURE # Si lo defines en settings
            )
            response_message = llm_response.response_content
        except Exception as e:
            print(f"❌ Error al llamar al LLM: {e}")
            response_message = "Lo siento, estoy teniendo problemas para responder en este momento. Por favor, inténtalo de nuevo más tarde."

        # 5. Devolver la respuesta encapsulada
        return ChatResponse(response_message=response_message)