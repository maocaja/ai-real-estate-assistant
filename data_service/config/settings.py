import os

class Settings:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "proyectos_colombia_es.json.json")  # nombre actual
    PROJECT_RESULTS_LIMIT = 50

settings = Settings()
