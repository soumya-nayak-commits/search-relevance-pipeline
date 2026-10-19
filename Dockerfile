FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app.py .

# The embedding index is not baked into the image. Build it once (python -m src.build_index)
# and mount it:  docker run -p 8000:8000 -v $(pwd)/data/index:/app/data/index search-api
ENV INDEX_DIR=/app/data/index
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
