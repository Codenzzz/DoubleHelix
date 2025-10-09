FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Ensure backend is in Python path
ENV PYTHONPATH=/app

EXPOSE 8000

# Launch from /app/backend but import backend.main:app
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
