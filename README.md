# Asistente de Bienes Raíces con IA (AI Real Estate Assistant)

Este proyecto implementa un asistente de inteligencia artificial para el sector de bienes raíces, diseñado para ayudar a los usuarios a encontrar propiedades en Colombia mediante el uso de procesamiento de lenguaje natural y búsqueda semántica. La arquitectura está compuesta por microservicios interconectados, cada uno con una responsabilidad específica.

## Arquitectura del Proyecto

El sistema se compone de tres microservicios principales, construidos con FastAPI:

1.  **`data-service` (Servicio de Datos):**
    * **Responsabilidad:** Sirve como la fuente de verdad para la información de los proyectos inmobiliarios. Carga un dataset de proyectos (actualmente desde un archivo JSON) y expone una API para consultar, filtrar y obtener detalles de los proyectos.
    * **Endpoints Clave:**
        * `GET /projects`: Lista y filtra proyectos por diversas características.
        * `GET /projects/{project_id}`: Obtiene detalles de un proyecto específico por su ID.
        * `GET /projects/amenities_data`: Proporciona datos de amenidades de proyectos en un formato optimizado para el `embedding-worker`.
    * **Tecnologías:** FastAPI, Pandas.

2.  **`embedding-worker` (Procesador de Embeddings):**
    * **Responsabilidad:** Transforma las descripciones de las amenidades de los proyectos en representaciones numéricas (`embeddings`) que permiten la búsqueda de similitud semántica. Utiliza un índice FAISS para realizar búsquedas eficientes.
    * **Dependencia:** Consume datos del `data-service` a través de su API.
    * **Endpoints Clave:**
        * `POST /search`: Realiza búsquedas de similitud semántica basadas en texto.
        * `GET /health`: Verifica el estado del servicio (modelo cargado, índice construido).
        * `POST /rebuild-index`: Permite forzar la reconstrucción del índice en segundo plano.
    * **Tecnologías:** FastAPI, `sentence-transformers`, `FAISS`.

3.  **`chat-api` (API de Chat):**
    * **Responsabilidad:** Es el punto de entrada para las interacciones del usuario. Orquesta las llamadas a los otros microservicios y a un Large Language Model (LLM) para entender las preguntas del usuario, recuperar información relevante y generar respuestas conversacionales.
    * **Dependencia:** Se comunica con el `data-service`, el `embedding-worker` y un proveedor de LLM (ej. OpenAI, Google Gemini).
    * **Endpoints Clave:**
        * `POST /chat`: Recibe mensajes de usuario y devuelve respuestas del asistente.
        * `GET /health`: Comprueba el estado de este servicio y sus dependencias.
    * **Tecnologías:** FastAPI, `httpx`, `openai` (o cliente para otro LLM).

## Configuración del Entorno y Ejecución

Sigue estos pasos para configurar tu entorno e iniciar todos los servicios.

### **Prerrequisitos**

* Python 3.9+
* `pip` (administrador de paquetes de Python)

### **1. Clonar el Repositorio**

```bash
git clone <URL_DE_TU_REPOSITORIO>
cd ai-real-estate-assistant



2. Crear y Activar el Entorno Virtual
Es buena práctica usar un entorno virtual para gestionar las dependencias del proyecto.

python3 -m venv .venv
source .venv/bin/activate  # En Linux/macOS
# .venv\Scripts\activate   # En Windows

3. Instalar Dependencias
Instala todas las librerías necesarias para los tres servicios.

pip install -r requirements.txt

4. Preparar el Dataset
Asegúrate de tener tu archivo de datos JSON en la ubicación esperada. Por defecto, el data-service espera un archivo llamado proyectos_colombia_es.json (o el nombre que hayas configurado en data_service/config/settings.py) dentro de la carpeta data/ en la raíz del proyecto.


5. Configurar Variables de Entorno (API Keys del LLM)
Para que el chat-api funcione, necesita una API Key para el modelo de lenguaje grande (LLM).

Crea un archivo llamado .env en la raíz de tu proyecto (al mismo nivel que data_service/, embedding_worker/, chat_api/) y añade tu API Key:

Fragmento de código

# .env file
OPENAI_API_KEY="sk-TU_CLAVE_SECRETA_AQUI"

6. Ejecutar los Microservicios
Necesitarás tres terminales separadas para ejecutar los servicios simultáneamente.

Terminal 1: data-service
Bash

# Asegúrate de estar en la raíz del proyecto y con el entorno virtual activado
uvicorn data_service.main:app --reload --port 8001
Verifica que la salida muestre ✅ Data loaded... y que el servicio esté corriendo en http://127.0.0.1:8001.

Terminal 2: embedding-worker
Bash

# Abre una nueva terminal, ve a la raíz del proyecto y activa el entorno virtual
uvicorn embedding_worker.main:app --reload --port 8002
Este servicio descargará el modelo de embeddings la primera vez (puede tardar un poco) y construirá el índice FAISS. Verifica mensajes como ✅ FAISS index built/refreshed successfully... y que esté corriendo en http://127.0.0.1:8002.

Terminal 3: chat-api
Bash

# Abre una tercera terminal, ve a la raíz del proyecto y activa el entorno virtual
uvicorn chat_api.main:app --reload --port 8003
Verifica que el servicio inicie correctamente en http://127.0.0.1:8003. Es posible que veas una advertencia si tu API Key no está configurada correctamente.

7. Probar el Asistente
Una vez que los tres servicios estén corriendo, puedes interactuar con el chat-api a través de su documentación interactiva.

Abre tu navegador y ve a: http://127.0.0.1:8003/docs

Busca el endpoint POST /chat.

Haz clic en "Try it out".

En el campo "Request body", introduce tu mensaje. Por ejemplo:

JSON

{
  "current_message": "¿Qué proyectos de apartamentos hay en Bogotá?",
  "conversation_history": []
}