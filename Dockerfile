FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data and HuggingFace model
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "from transformers import CLIPProcessor, CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Copy application code
COPY backend/ ./

# Environment variables will be passed at runtime

# Command to run the FastAPI application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT