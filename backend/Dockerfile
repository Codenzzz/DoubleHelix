FROM python:3.11-slim

# Set the working directory explicitly to /app/backend
WORKDIR /app

# Install dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything into /app
COPY . .

EXPOSE 8000

# ✅ ENTRYPOINT FIX: ensure correct import path
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

