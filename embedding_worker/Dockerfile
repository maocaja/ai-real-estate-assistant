FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar curl para pruebas de red
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "embedding_worker.main:app", "--host", "0.0.0.0", "--port", "8000"] 