from typing import List, Dict, Any, Optional, Tuple

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
        self.system_message_content = (
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
            "**Cuando la respuesta final contenga una lista de proyectos, asegúrate de que el LLM los devuelva en un formato JSON con las claves 'response_message' (el mensaje para el usuario) y 'recommended_projects' (una lista de objetos Project).**"
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
        all_recommended_projects: List[Project] = [] # Lista para acumular proyectos recomendados

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
                    # El LLM ha respondido con contenido final.
                    # Intentar parsear como JSON si se espera un ChatResponse estructurado,
                    # de lo contrario, tratar como texto plano.
                    message_raw = llm_response_or_tool_call.response_content
                    response_message = str(message_raw) # Default to raw string
                    projects_from_llm = []

                    if isinstance(message_raw, str):
                        try:
                            parsed_content = json.loads(message_raw)
                            if isinstance(parsed_content, dict):
                                response_message = parsed_content.get("response_message", str(message_raw))
                                raw_projects = parsed_content.get("recommended_projects", [])
                                # Convertir raw_projects a objetos Project si son diccionarios
                                try:
                                    projects_from_llm = [Project(**proj) for proj in raw_projects]
                                except Exception as e:
                                    print(f"Error al convertir proyectos del LLM a Project: {e}")
                                    projects_from_llm = []
                        except json.JSONDecodeError:
                            # Not a JSON string, treat as plain text
                            pass
                    elif isinstance(message_raw, dict):
                        # If LLM directly returns a dict (e.g., from a structured response mode)
                        response_message = message_raw.get("response_message", str(message_raw))
                        raw_projects = message_raw.get("recommended_projects", [])
                        try:
                            projects_from_llm = [Project(**proj) for proj in raw_projects]
                        except Exception as e:
                            print(f"Error al convertir proyectos del LLM a Project (dict): {e}")
                            projects_from_llm = []

                    # Combinar proyectos de la respuesta directa del LLM con los acumulados de las herramientas
                    # Usar un conjunto para evitar duplicados si los IDs de proyecto son únicos
                    unique_projects = {p.id: p for p in all_recommended_projects + projects_from_llm}.values()
                    final_recommended_projects = list(unique_projects)

                    return ChatResponse(
                        response_message=response_message,
                        recommended_projects=final_recommended_projects
                    )

                elif llm_response_or_tool_call.tool_calls:
                    tool_call_count += 1
                    # IMPORTANT: Add the assistant's message with ALL tool_calls to the history FIRST.
                    messages_for_llm.append(Message(
                        role="assistant",
                        tool_calls=llm_response_or_tool_call.tool_calls
                    ))

                    # Process ALL tool calls suggested by the LLM
                    tool_outputs = []
                    for tool_call_dict in llm_response_or_tool_call.tool_calls:
                        function_name = tool_call_dict["function"]["name"]
                        function_args_str = tool_call_dict["function"]["arguments"]
                        tool_call_id = tool_call_dict["id"]

                        try:
                            function_args = json.loads(function_args_str)
                        except json.JSONDecodeError:
                            error_message = f"Error: LLM returned invalid function arguments for {function_name}: {function_args_str}"
                            tool_outputs.append(Message(role="tool", content=error_message, name=function_name, tool_call_id=tool_call_id))
                            continue

                        # Default values for tool response
                        llm_tool_response_string = "Lo siento, no pude ejecutar la herramienta."
                        current_tool_projects: List[Project] = []

                        try:
                            # Attempt to execute the function from the appropriate client
                            if hasattr(self.data_service_client, function_name):
                                method = getattr(self.data_service_client, function_name)
                                result = await method(**function_args)
                                llm_tool_response_string, current_tool_projects = self._format_data_service_response(function_name, result)

                            elif hasattr(self.embedding_service_client, function_name):
                                method = getattr(self.embedding_service_client, function_name)
                                result = await method(**function_args)
                                llm_tool_response_string, current_tool_projects = self._format_embedding_service_response(function_name, result)

                            elif hasattr(self.llm_client, function_name):
                                llm_tool_response_string = "No se permite llamar a funciones internas del LLM."
                                current_tool_projects = []

                            else:
                                llm_tool_response_string = f"Error: The function {function_name} does not exist or is not accessible."
                                current_tool_projects = []

                        except Exception as e:
                            print(f"❌ Error during tool execution for {function_name}: {e}")
                            llm_tool_response_string = f"Error al ejecutar la herramienta {function_name}: {str(e)}"
                            current_tool_projects = []

                        print(f"✅ Tool '{function_name}' responded: {llm_tool_response_string[:200]}...")

                        # Add projects from this tool call to the accumulated list
                        all_recommended_projects.extend(current_tool_projects)

                        # Add EACH tool's response as a separate tool message
                        tool_outputs.append(Message(
                            role="tool",
                            name=function_name,
                            content=llm_tool_response_string,
                            tool_call_id=tool_call_id
                        ))

                    # After processing ALL tool calls in the current LLM turn,
                    # extend the messages list with all their outputs.
                    messages_for_llm.extend(tool_outputs)

                    # Continue the loop so the LLM can use the combined tool outputs to finalize the reply
                    continue

                else:
                    print("❌ LLM did not return text or tool calls.")
                    return ChatResponse(response_message="Lo siento, ocurrió un error al procesar tu solicitud.", recommended_projects=all_recommended_projects)

            except Exception as e:
                print(f"❌ Error during message handling or tool execution: {e}")
                return ChatResponse(response_message="Lo siento, estoy teniendo problemas para responder en este momento. Por favor, intenta más tarde.", recommended_projects=all_recommended_projects)

        # If maximum tool call attempts are exceeded without resolution
        print("⚠️ Maximum tool calls exceeded or LLM did not finalize response.")
        return ChatResponse(response_message="Lo siento, no pude completar tu solicitud después de varios intentos", recommended_projects=all_recommended_projects)

    # --- Métodos Auxiliares para Formatear Respuestas de Herramientas ---
    # Estos métodos convierten los objetos Python obtenidos de los servicios en texto
    # que el LLM pueda entender y usar para generar su respuesta, y también devuelven
    # la lista de proyectos relevantes.
    def _format_data_service_response(self, function_name: str, data: Any) -> Tuple[str, List[Project]]:
        projects_to_return: List[Project] = []
        response_string = ""

        if function_name == "get_filtered_projects":
            if not data:
                response_string = "No se encontraron proyectos con los criterios especificados."
            else:
                try:
                    projects_to_return = []
                    for p in data[:5]:
                        if isinstance(p, Project):
                            projects_to_return.append(p)
                        elif hasattr(p, "model_dump"):
                            # Viene de otro modelo Pydantic (como data_service.models.project.Project)
                            projects_to_return.append(Project(**p.model_dump()))
                        elif isinstance(p, dict):
                            projects_to_return.append(Project(**p))
                        else:
                            print(f"❗ Tipo de dato inesperado: {type(p)}")
                except Exception as e:
                    print(f"❌ Error al convertir datos a Project en get_filtered_projects: {e}")
                    projects_to_return = []
                if len(data) > 5:
                    response_string += "\n(Y posiblemente más proyectos con estos criterios.)"

        elif function_name == "get_project_by_id":
            if not data:
                response_string = "No se encontró el proyecto con el ID especificado."
            else:
                project_data_as_dict = None
                if isinstance(data, dict):
                    project_data_as_dict = data
                elif hasattr(data, 'dict'): # Para modelos Pydantic v1
                    project_data_as_dict = data.dict()
                elif hasattr(data, 'model_dump'): # Para modelos Pydantic v2
                    project_data_as_dict = data.model_dump()
                else:
                    print(f"Advertencia: Tipo de dato inesperado para Project en get_project_by_id: {type(data)}. Intentando serializar a JSON.")
                    try:
                        project_data_as_dict = json.loads(json.dumps(data))
                    except Exception as e:
                        print(f"Error al serializar tipo inesperado a JSON: {e}")
                        project_data_as_dict = {}

                project_instance: Optional[Project] = None
                if project_data_as_dict:
                    try:
                        project_instance = Project(**project_data_as_dict)
                    except Exception as e:
                        print(f"Error al crear Project desde diccionario en get_project_by_id: {e}")

                if project_instance:
                    projects_to_return = [project_instance]
                    response_string = ""
                else:
                    response_string = "No se pudo obtener la información completa del proyecto."
        else:
            response_string = json.dumps(data) # Por defecto, serializar a JSON si no hay formato específico

        return response_string, projects_to_return

    def _format_embedding_service_response(self, function_name: str, data: Any) -> Tuple[str, List[Project]]:
        projects_to_return: List[Project] = []
        response_string = ""

        if function_name == "search_similar_projects":
            if not data:
                response_string = "No se encontraron proyectos similares con los criterios de búsqueda semántica."
            else:
                response_string = f"Se encontraron proyectos con IDs similares: {', '.join(data)}. El LLM puede preguntar por detalles de estos IDs si es necesario."
        else:
            response_string = json.dumps(data)

        return response_string, projects_to_return

    def _format_financial_assessment_response(self, data: Dict[str, Any]) -> Tuple[str, List[Project]]:
        """
        Formatea la respuesta del cálculo financiero y de inversión para que el LLM la entienda.
        No devuelve proyectos, por lo que la lista de proyectos estará vacía.
        """
        response_string = ""
        if "error" in data:
            response_string = f"Error en el cálculo financiero: {data['error']}"
        else:
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
            response_string = "\n".join(response_parts)

        return response_string, [] # No hay proyectos en la respuesta financiera
