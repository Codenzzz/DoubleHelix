FROM python:3.11-slim

WORKDIR /app

# Copy reqs and force a fresh install via cache-bust
COPY requirements.txt ./
ARG CACHEBUST=20251008-2
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
