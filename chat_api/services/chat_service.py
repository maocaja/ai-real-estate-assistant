from typing import List, Dict, Any, Optional
import json

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
        # Define el mensaje del sistema para guiar al LLM.
        self.system_message_content = (
            "Eres un asistente de bienes raíces experto y amigable especializado en proyectos de vivienda "
            "en Colombia. Tu objetivo es ayudar a los usuarios a encontrar el proyecto ideal "
            "proporcionando información relevante, filtrando proyectos y respondiendo preguntas. "
            "Siempre responde de manera concisa y útil, ofreciendo solo la información solicitada. "
            "Si un usuario pide buscar un proyecto con ciertas características, primero busca proyectos "
            "usando las herramientas disponibles y luego presenta la información relevante de esos proyectos. "
            "No asumas información no proporcionada por las herramientas. Si no encuentras información "
            "relevante, dilo. Si te preguntan por un país diferente a Colombia, indica que tu expertise es Colombia."
            "IMPORTANTE: Utiliza las herramientas disponibles para responder a las preguntas de los usuarios."
            "Si la pregunta del usuario es una búsqueda de proyectos, DEBES usar la herramienta `get_filtered_projects`."
            "Si la pregunta del usuario es una descripción de amenidades que desea, DEBES usar la herramienta `search_similar_projects`."
            "Si el usuario pregunta por detalles de un proyecto específico por su ID, DEBES usar la herramienta `get_project_by_id`."
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_filtered_projects",
                    "description": "Obtiene una lista de proyectos inmobiliarios filtrados por ciudad, tipo de propiedad, rango de precios, número de habitaciones, baños, área o nombre del proyecto. Usar cuando el usuario especifica criterios de búsqueda estructurados. Los precios deben ser en COP.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "La ciudad en la que se buscan proyectos (ej. 'Bogotá', 'Medellín')."
                            },
                            "property_type": {
                                "type": "string",
                                "description": "El tipo de propiedad (ej. 'Apartamento', 'Casa')."
                            },
                            "min_price": {
                                "type": "number",
                                "description": "El precio mínimo en COP."
                            },
                            "max_price": {
                                "type": "number",
                                "description": "El precio máximo en COP."
                            },
                            "min_bedrooms": {
                                "type": "integer",
                                "description": "Número mínimo de habitaciones."
                            },
                            "min_bathrooms": {
                                "type": "integer",
                                "description": "Número mínimo de baños."
                            },
                            "min_area_sqm": {
                                "type": "number",
                                "description": "Área mínima en metros cuadrados."
                            },
                            "recommended_use": {
                                "type": "string",
                                "description": "Uso recomendado del proyecto (ej. 'Residencial', 'Comercial')."
                            },
                            "project_name": {
                                "type": "string",
                                "description": "Nombre específico o parcial del proyecto a buscar."
                            }
                        },
                        "required": [] # No hay parámetros obligatorios si se pueden combinar los filtros
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_similar_projects",
                    "description": "Busca proyectos inmobiliarios que son semánticamente similares a una descripción de texto dada, útil para encontrar proyectos basados en amenidades o características no estructuradas. Por ejemplo: 'proyectos con piscina y gimnasio' o 'que tengan buenas zonas verdes'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "La descripción de texto para buscar proyectos similares (ej. 'proyectos con zonas verdes para niños')."
                            },
                            "project_ids_subset": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Opcional. Una lista de IDs de proyectos para limitar la búsqueda de similitud dentro de ellos."
                            }
                        },
                        "required": ["query_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_project_by_id",
                    "description": "Obtiene los detalles completos de un proyecto específico dado su ID único. Usar cuando el usuario pregunta directamente por un proyecto por su identificador.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_id": {
                                "type": "string",
                                "description": "El ID único del proyecto (ej. 'PROYECTO_123')."
                            }
                        },
                        "required": ["project_id"]
                    }
                }
            }
        ]

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        """
        Handles an incoming user message by orchestrating LLM interactions and tool (function) calls using OpenAI's Function Calling.

        This method:
        - Prepares the full message history for the LLM.
        - Calls the LLM to generate a response.
        - If the LLM suggests function calls, executes them and loops back with the tool output.
        - Stops when a final message is produced or when the tool call limit is reached.

        Args:
            request (ChatRequest): Contains the current user message and conversation history.

        Returns:
            ChatResponse: The response to be sent back to the user.
        """
        messages_for_llm: List[Message] = []

        # 1. Add the system prompt (guidance for the LLM)
        messages_for_llm.append(Message(role="system", content=self.system_message_content))

        # 2. Add recent conversation history (for context preservation)
        history_to_include = request.conversation_history[-settings.MAX_CONVERSATION_HISTORY:]
        messages_for_llm.extend(history_to_include)

        # 3. Add the current user message
        messages_for_llm.append(Message(role="user", content=request.current_message))

        # --- Tool Calling Loop ---
        MAX_TOOL_CALLS = 3
        tool_call_count = 0

        while tool_call_count < MAX_TOOL_CALLS:
            try:
                # Call the LLM to get a response or a tool call suggestion
                llm_response_or_tool_call: LLMResponse = await self.llm_client.get_chat_completion(
                    messages=messages_for_llm,
                    tools=self.tools,
                    temperature=settings.OPENAI_LLM_TEMPERATURE
                )

                if llm_response_or_tool_call.response_content:
                    # The LLM returned a final response directly — return it
                    return ChatResponse(response_message=llm_response_or_tool_call.response_content)
                
                elif llm_response_or_tool_call.tool_calls:
                    # The LLM suggested one or more tool (function) calls
                    tool_call_count += 1

                    # --- Important: Add assistant message with tool_calls before executing any tool ---
                    messages_for_llm.append(Message(
                        role="assistant",
                        tool_calls=llm_response_or_tool_call.tool_calls
                    ))

                    # Currently, only process the first tool call
                    tool_call = llm_response_or_tool_call.tool_calls[0]
                    function_name = tool_call["function"]["name"]
                    function_args_str = tool_call["function"]["arguments"]

                    try:
                        # Attempt to parse function arguments from JSON string
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError:
                        # Invalid argument format — report back to LLM
                        error_message = f"Error: LLM returned invalid function arguments for {function_name}: {function_args_str}"
                        messages_for_llm.append(Message(role="tool", content=error_message, name=function_name))
                        print(error_message)
                        continue  # Continue to next loop iteration

                    print(f"✨ LLM suggested calling: {function_name} with args: {function_args}")

                    function_response = "Lo siento, no pude ejecutar la herramienta."

                    # Attempt to execute the function from the appropriate client
                    if hasattr(self.data_service_client, function_name):
                        method = getattr(self.data_service_client, function_name)
                        result = await method(**function_args)
                        function_response = self._format_data_service_response(function_name, result)

                    elif hasattr(self.embedding_service_client, function_name):
                        method = getattr(self.embedding_service_client, function_name)
                        result = await method(**function_args)
                        function_response = self._format_embedding_service_response(function_name, result)

                    elif hasattr(self.llm_client, function_name):
                        function_response = "No se permite llamar a funciones internas del LLM."

                    else:
                        function_response = f"Error: The function {function_name} does not exist or is not accessible."

                    print(f"✅ Tool '{function_name}' responded: {function_response[:200]}...")

                    # Add the tool's response so the LLM can continue the conversation
                    messages_for_llm.append(Message(
                        role="tool",
                        name=function_name,
                        content=function_response,
                        tool_call_id=tool_call["id"]
                    ))

                    # Continue the loop so the LLM can use the tool output to finish the reply

                else:
                    # Unexpected: LLM did not return a response or tool call
                    print("❌ LLM did not return text or tool calls.")
                    return ChatResponse(response_message="Sorry, something went wrong while processing your request.")

            except Exception as e:
                # General error while calling the LLM or tool
                print(f"❌ Error during message handling or tool execution: {e}")
                return ChatResponse(response_message="Sorry, I'm having trouble responding right now. Please try again later.")
        
        # If maximum tool call attempts are exceeded without resolution
        print("⚠️ Maximum tool calls exceeded or LLM did not finalize response.")
        return ChatResponse(response_message="Sorry, I couldn't complete your request after several attempts.")

    # --- Métodos Auxiliares para Formatear Respuestas de Herramientas ---
    # Estos métodos convierten los objetos Python obtenidos de los servicios en texto
    # que el LLM pueda entender y usar para generar su respuesta.
    def _format_data_service_response(self, function_name: str, data: Any) -> str:
        if function_name == "get_filtered_projects":
            if not data:
                return "No se encontraron proyectos con los criterios especificados."
            
            # Formatear una lista de proyectos para el LLM
            formatted_projects = []
            for project in data[:5]: # Limitar para no saturar el prompt
                amenities = project.amenities if project.amenities else "No especificado"
                formatted_projects.append(
                    f"- {project.project_name}: Ubicado en {project.city}, {project.type_property}. "
                    f"Precio: ${project.price_min if project.price_max else 'N/A'}. "
                    f"Habitaciones: {project.bedrooms_min if project.bedrooms_min else 'N/A'}. "
                    f"Amenidades: {amenities}."
                )
            return "Se encontraron los siguientes proyectos:\n" + "\n".join(formatted_projects) + "\n(Y posiblemente más proyectos con estos criterios.)"

        elif function_name == "get_project_by_id":
            if not data:
                return "No se encontró el proyecto con el ID especificado."
            amenities_text = data.amenities if data.amenities else "No especificado"

            return (
                f"Detalles del proyecto '{data.project_name}' (ID: {data.id}):\n" # Use project_name
                f"Ubicación: {data.city}, {data.zone}\n" # Use project.city and project.zone
                f"Tipo: {data.type_property if data.type_property else 'N/A'}\n" # Use project.type_property
                f"Precios: Desde ${data.price_min if data.price_min else 'N/A'} hasta ${data.price_max if data.price_max else 'N/A'}\n" # Use price_min, price_max
                f"Habitaciones: {data.bedrooms_min if data.bedrooms_min else 'N/A'}" # Use bedrooms_min
                # Your Project model doesn't have bedrooms_max, so either add it or remove this part
                # If you want to show a range, you'll need bedrooms_max in your model and JSON.
                # For now, I'll remove bedrooms_max and bathrooms_max since they're not in your model
                # or you'll need to define them with aliases in Project.
                f"Baños: {data.bedrooms_min if data.bedrooms_min else 'N/A'}\n" # Assuming you add bedrooms_min to Project model
                # Your Project model doesn't have area_maxima_m2_hasta, use min_area_sqm if that's what you have
                f"Área: {data.area_sqm if data.area_sqm else 'N/A'} m²\n" # Assuming you add min_area_sqm to Project model
                f"Estado: {data.status if data.status else 'N/A'}\n" # Use project.status
                #f"Fecha de Entrega: {data.delivery_date_estimated if project.delivery_date_estimated else 'N/A'}\n" # Use delivery_date_estimated
                f"Amenidades: {amenities_text}\n" # Use amenities_text from above
                f"URL: {data.image_url if data.image_url else 'N/A'}" # Using image_url as closest to url_inmueble
            )
        return json.dumps(data) # Por defecto, serializar a JSON si no hay formato específico

    def _format_embedding_service_response(self, function_name: str, data: Any) -> str:
        if function_name == "search_similar_projects":
            if not data:
                return "No se encontraron proyectos similares con los criterios de búsqueda semántica."
            
            # Aquí podrías opcionalmente llamar al data_service para obtener los nombres de estos IDs
            # Pero para mantenerlo simple, devolvemos los IDs. El LLM puede pedir detalles si lo necesita.
            return f"Se encontraron proyectos con IDs similares: {', '.join(data)}. El LLM puede preguntar por detalles de estos IDs si es necesario."
        return json.dumps(data)