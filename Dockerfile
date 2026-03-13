# syntax=docker/dockerfile:1

FROM python:3.11.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install requirements from mlflow artifacts
COPY app/model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia solo il codice necessario
COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
