# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the backend requirements file and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all of the backend code into the container
COPY backend/ .

# Expose port 8000 (Railway will supply the PORT environment variable at runtime)
EXPOSE 8000

# Start the FastAPI app; if PORT isn’t set, default to 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
