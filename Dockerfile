FROM python:3.10-slim

# System deps for unstructured
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY data data
COPY vectorstore vectorstore

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
