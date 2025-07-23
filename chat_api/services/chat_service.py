from typing import List, Dict, Any, Optional
import json

from chat_api.config.settings import settings
from chat_api.models.chat_models import Message, ChatRequest, ChatResponse, LLMResponse, Project
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
        """
            IMPORTANTE: Responde SIEMPRE en formato JSON puro. El formato esperado es:
            RESPONDE EXCLUSIVAMENTE EN FORMATO JSON.
            {
            "response_message": "string",
            "recommended_projects": [
                {
                "id": "string",
                "nombre_proyecto": "string",
                "tipo": "string",
                "ciudad": "string",
                "precio_minimo_desde": 1000000,
                "precio_maximo_hasta": 2000000,
                "habitaciones_minimo_desde": 2,
                "habitaciones_maximo_hasta": 4,
                "banios_minimo_desde": 2,
                "banios_maximo_hasta": 3,
                "area_minima_m2_desde": 60,
                "area_maxima_m2_hasta": 80,
                "tipo_uso_recomendado": "Residencial",
                "descripcion": "Hermoso proyecto en zona norte de Bogotá",
                "url_proyecto": "https://...",
                "url_imagen": "https://...",
                "latitud": 4.711,
                "longitud": -74.0721,
                "amenidades": ["Piscina", "Gimnasio"]
                }
            ]
            }
            ⚠️ SI NO HAY PROYECTOS QUE CUMPLAN CON LO SOLICITADO, DEBES DEVOLVER IGUALMENTE EL JSON CON `recommended_projects: []`.

            ⚠️ NO ENVUELVAS NUNCA LA RESPUESTA EN BLOQUES DE CÓDIGO COMO ```json.

            ⚠️ SI DEVUELVES TEXTO PLANO, NO SE MOSTRARÁ CORRECTAMENTE AL USUARIO.

            NO uses markdown (```json). NO respondas con texto plano. Solo responde con ese JSON.
            Si haces un análisis financiero usando la herramienta `get_financial_assessment`, DEBES incluir el análisis completo dentro del campo `response_message` del JSON final. **No repitas los detalles completos de los proyectos aquí si ya los incluiste en el campo `recommended_projects`.** Puedes referenciarlos brevemente si es necesario, pero no dupliques datos.
            El campo `response_message` debe ser un texto completamente natural, redactado como si fuera parte de una conversación fluida. No uses saltos de línea (\n), listas con guiones (-), numeraciones, ni formato Markdown como negritas (**). Describe la información de manera continua y amigable. 
            Mantén los valores importantes como precios (por ejemplo: $180,000,000 COP), porcentajes (por ejemplo: 12%) y unidades (como m² o número de habitaciones) exactamente como son para facilitar la visualización por parte del usuario.
            Al final de tu respuesta DEBES incluir el análisis completo dentro del campo `response_message` del JSON final, si has presentado uno o varios proyectos, concluye con una frase amable como: "Si necesitas más información o buscas algo más específico, házmelo saber o algo mimilar dependiendo el contexto."
            """ +
            "Eres un asistente de bienes raíces experto y amigable especializado en proyectos de vivienda "
            "en Colombia. Tu objetivo es ayudar a los usuarios a encontrar el proyecto ideal "
            "proporcionando información relevante, filtrando proyectos y respondiendo preguntas. "
            "Siempre responde de manera concisa y útil, ofreciendo solo la información solicitada. "
            "Si un usuario pide buscar un proyecto con ciertas características, primero busca proyectos "
            "usando las herramientas disponibles y luego presenta la información relevante de esos proyectos. "
            "Al presentar varios proyectos, resume su información clave (nombre, ciudad, tipo de propiedad, precio min/max) de forma que el usuario no necesite pedir los detalles de cada ID. "
            "No asumas información no proporcionada por las herramientas. Si no encuentras información "
            "relevante, dilo. Si te preguntan por un país diferente a Colombia, indica que tu expertise es Colombia."
            "IMPORTANTE: Utiliza las herramientas disponibles para responder a las preguntas de los usuarios."
            "Si la pregunta del usuario es una búsqueda de proyectos, DEBES usar la herramienta `get_filtered_projects`."
            "Si la pregunta del usuario es una descripción de amenidades que desea, DEBES usar la herramienta `search_similar_projects`."
            "Si el usuario pregunta por detalles de un proyecto específico por su ID, DEBES usar la herramienta `get_project_by_id`."
            "**Si el usuario pregunta sobre financiación, cuotas, créditos, retorno de inversión (ROI), flujo de caja o cómo adquirir un proyecto, DEBES usar la herramienta `get_financial_assessment`.** "
            "**Para usar `get_financial_assessment`, necesitas el precio del proyecto, los ingresos mensuales del usuario y el plazo deseado del préstamo en años.** "
            "**Si el usuario pregunta por análisis de inversión (ROI o flujo de caja), también necesitarás preguntar por el ingreso de alquiler mensual esperado, la tasa de apreciación anual esperada del inmueble y el horizonte de inversión en años.** "
            "**Si te falta alguno de estos datos para el cálculo que solicita el usuario, pregúntale de forma educada para poder realizar el análisis completo.** "
            "**Si el usuario ya ha preguntado por un proyecto, intenta usar el precio de ese proyecto para la estimación financiera.**"
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
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financial_assessment",
                    "description": "Calcula una estimación de cuota mensual, valor total a pagar, y un análisis básico de inversión (ROI y flujo de caja) para un proyecto de vivienda. Utilizar cuando el usuario pregunta sobre financiación, cuotas, créditos, retorno de inversión, flujo de caja o cómo adquirir un proyecto.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_price": {
                                "type": "number",
                                "description": "El precio total del proyecto para el cálculo en COP. Es un valor en COP."
                            },
                            "monthly_income": {
                                "type": "number",
                                "description": "Los ingresos mensuales netos del usuario en COP."
                            },
                            "loan_term_years": {
                                "type": "integer",
                                "description": "El plazo deseado del préstamo en años (ej. 15, 20, 30)."
                            },
                            "interest_rate_annual": {
                                "type": "number",
                                "description": "Opcional. La tasa de interés anual en porcentaje (ej. 12 para 12%). Si no se especifica, usa una tasa de interés predeterminada razonable para Colombia (ej. 12.0)."
                            },
                            "expected_rental_income_monthly": { # <-- NUEVO CAMPO
                                "type": "number",
                                "description": "Opcional. El ingreso de alquiler mensual estimado para el proyecto en COP. Necesario para el cálculo de flujo de caja y ROI."
                            },
                            "expected_annual_appreciation_rate": { # <-- NUEVO CAMPO
                                "type": "number",
                                "description": "Opcional. La tasa de apreciación anual esperada del valor del inmueble en porcentaje (ej. 5 para 5%). Necesario para el cálculo de ROI."
                            },
                            "annual_operating_costs_percentage": { # <-- NUEVO CAMPO
                                "type": "number",
                                "description": "Opcional. Porcentaje anual del precio del proyecto destinado a costos operativos y de mantenimiento (ej. 1 para 1%). Necesario para el cálculo de flujo de caja y ROI. Por defecto 1.0% si no se especifica."
                            },
                            "investment_horizon_years": { # <-- NUEVO CAMPO
                                "type": "integer",
                                "description": "Opcional. El horizonte de tiempo en años para el cálculo del ROI y flujo de caja (ej. 5, 10). Por defecto 10 años si no se especifica."
                            }
                        },
                        "required": ["project_price", "monthly_income", "loan_term_years"] # Los nuevos campos son opcionales por ahora
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
                    message_raw = llm_response_or_tool_call.response_content
                    if isinstance(message_raw, str):
                        try:
                            parsed = json.loads(message_raw)
                        except json.JSONDecodeError:
                            return ChatResponse(response_message=message_raw, recommended_projects=[])
                    elif isinstance(message_raw, dict):
                        parsed = message_raw
                    else:
                        return ChatResponse(response_message=str(message_raw), recommended_projects=[])

                    response_message = parsed.get("response_message", str(message_raw))
                    raw_projects = parsed.get("recommended_projects", [])
                    projects = [Project(**proj) for proj in raw_projects]
                    return ChatResponse(
                        response_message=response_message,
                        recommended_projects=projects
                    )

                elif llm_response_or_tool_call.tool_calls:
                    tool_call_count += 1
                    # IMPORTANT: Add the assistant's message with ALL tool_calls to the history FIRST.
                    # Ensure tool_calls are converted to dictionaries for the OpenAI API.
                    messages_for_llm.append(Message(
                        role="assistant",
                        tool_calls=llm_response_or_tool_call.tool_calls
                    ))

                    # Process ALL tool calls suggested by the LLM
                    tool_outputs = []
                    for tool_call_dict in llm_response_or_tool_call.tool_calls:
                        # Convert the Pydantic ToolCall object to a dictionary
                        function_name = tool_call_dict["function"]["name"]
                        function_args_str = tool_call_dict["function"]["arguments"]
                        tool_call_id = tool_call_dict["id"] # Get the tool_call_id from the dict

                        try:
                            # Attempt to parse function arguments from JSON string
                            function_args = json.loads(function_args_str)
                        except json.JSONDecodeError:
                            # Invalid argument format – report back to LLM
                            error_message = f"Error: LLM returned invalid function arguments for {function_name}: {function_args_str}"
                            # Even for an error, we must provide a tool response for this tool_call_id
                            tool_outputs.append(Message(role="tool", content=error_message, name=function_name, tool_call_id=tool_call_id))
                            continue  # Skip to the next tool_call in the list

                        function_response_content = "Lo siento, no pude ejecutar la herramienta."

                        try:
                            # Attempt to execute the function from the appropriate client
                            if hasattr(self.data_service_client, function_name):
                                method = getattr(self.data_service_client, function_name)
                                result = await method(**function_args)
                                if function_name == "get_financial_assessment":
                                    function_response_content = self._format_financial_assessment_response(result)
                                else:
                                    function_response_content = self._format_data_service_response(function_name, result)

                            elif hasattr(self.embedding_service_client, function_name):
                                method = getattr(self.embedding_service_client, function_name)
                                result = await method(**function_args)
                                function_response_content = self._format_embedding_service_response(function_name, result)

                            elif hasattr(self.llm_client, function_name):
                                function_response_content = "No se permite llamar a funciones internas del LLM."

                            else:
                                function_response_content = f"Error: The function {function_name} does not exist or is not accessible."
                        except Exception as e:
                            # Catch any runtime errors during tool execution
                            print(f"❌ Error during tool execution for {function_name}: {e}")
                            function_response_content = f"Error al ejecutar la herramienta {function_name}: {str(e)}"

                        print(f"✅ Tool '{function_name}' responded: {function_response_content[:200]}...")

                        # Add EACH tool's response as a separate tool message
                        tool_outputs.append(Message(
                            role="tool",
                            name=function_name,
                            content=function_response_content,
                            tool_call_id=tool_call_id # <-- CRITICAL: Include the tool_call_id here
                        ))

                    # After processing ALL tool calls in the current LLM turn,
                    # extend the messages list with all their outputs.
                    messages_for_llm.extend(tool_outputs)

                    # Continue the loop so the LLM can use the combined tool outputs to finalize the reply
                    continue 

                else:
                    # If we reach here, the LLM didn't return a response or tool call. This shouldn't happen normally.
                    print("❌ LLM did not return text or tool calls.")
                    return ChatResponse(response_message="Lo siento, ocurrió un error al procesar tu solicitud.")

            except Exception as e:
                # General error while calling the LLM or tool
                print(f"❌ Error during message handling or tool execution: {e}")
                return ChatResponse(response_message="Lo siento, estoy teniendo problemas para responder en este momento. Por favor, intenta más tarde.")
        
        # If maximum tool call attempts are exceeded without resolution
        print("⚠️ Maximum tool calls exceeded or LLM did not finalize response.")
        return ChatResponse(response_message="Lo siento, no pude completar tu solicitud después de varios intentos")

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
                f"Área: {data.area_min_square if data.area_min_square else 'N/A'} m²\n" # Assuming you add min_area_sqm to Project model
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
    
    def _format_financial_assessment_response(self, data: Dict[str, Any]) -> str:
        """
        Formatea la respuesta del cálculo financiero y de inversión para que el LLM la entienda.
        """
        if "error" in data:
            return f"Error en el cálculo financiero: {data['error']}"

        # Formateo de números para mejor lectura
        def format_currency(value):
            return "{:,.0f}".format(value).replace(",", "_").replace("_", ",")

        response_parts = [
            f"Resultados de la estimación financiera para el proyecto de ${format_currency(data['project_price'])} COP:",
            f"- **Cuota inicial estimada (30%)**: ${format_currency(data['down_payment'])} COP",
            f"- **Monto del préstamo estimado (70%)**: ${format_currency(data['loan_amount'])} COP",
            f"- **Plazo del préstamo**: {data['loan_term_years']} años",
            f"- **Tasa de interés anual simulada**: {data['annual_interest_rate']}%",
            f"- **Cuota mensual estimada del préstamo**: ${format_currency(data['monthly_loan_payment'])} COP",
            f"- **Total pagado con intereses durante el plazo del préstamo**: ${format_currency(data['total_paid_with_interest'])} COP",
            f"- **Mensaje de asequibilidad**: {data['affordability_message']}"
        ]

        # Añadir información de inversión si está disponible
        if "roi_percentage" in data:
            response_parts.append("\n--- Análisis de Inversión ---")
            response_parts.append(f"- **Ingreso de alquiler mensual estimado**: ${format_currency(data['expected_rental_income_monthly'])} COP")
            response_parts.append(f"- **Tasa de apreciación anual esperada**: {data['expected_annual_appreciation_rate']}%")
            response_parts.append(f"- **Costos operativos anuales (% del valor del proyecto)**: {data['annual_operating_costs_percentage']}%")
            response_parts.append(f"- **Horizonte de inversión**: {data['investment_horizon_years']} años")
            response_parts.append(f"- **Valor futuro estimado del inmueble**: ${format_currency(data['future_property_value'])} COP")
            response_parts.append(f"- **Flujo de caja anual estimado**: ${format_currency(data['annual_cash_flow'])} COP")
            response_parts.append(f"- **Retorno de Inversión (ROI) estimado**: {data['roi_percentage']}%")
        elif "investment_analysis_message" in data:
            response_parts.append(f"\n--- Análisis de Inversión ---")
            response_parts.append(data['investment_analysis_message'])

        response_parts.append("\nEl LLM debe usar esta información para responder al usuario.")
        return "\n".join(response_parts)
