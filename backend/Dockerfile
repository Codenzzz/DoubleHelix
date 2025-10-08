FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

EXPOSE 8000

# 👇 FIXED: main.py is at root
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
