FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN pip install --no-cache-dir -e .

CMD ["uvicorn", "data_service.main:app", "--host", "0.0.0.0", "--port", "8000"] 