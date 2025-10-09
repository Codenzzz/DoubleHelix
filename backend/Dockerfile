FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Work from /app
WORKDIR /app

# Install dependencies first (layer cache)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repo into the image
COPY . .

# Make sure Python can import from /app
ENV PYTHONPATH=/app

# (optional) sanity check during build — fails early if imports break
RUN python -c "import importlib; importlib.import_module('backend.main'); print('import ok')"

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
