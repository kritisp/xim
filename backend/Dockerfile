FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# Copy data files the backend needs at runtime
COPY data/ ../data/
COPY data_pipeline/faiss_index.bin ../data_pipeline/faiss_index.bin
COPY data_pipeline/title_ids.json ../data_pipeline/title_ids.json

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
