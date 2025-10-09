FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy repo
COPY . .

ENV PYTHONPATH=/app

# Early check at build time (still useful)
RUN python -c "import importlib; importlib.import_module('backend.main'); print('import ok')"

# NEW: entrypoint that always runs bootcheck, then execs CMD or override
COPY backend/start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]

# Default command if Render doesn't override
CMD ["python","-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000","--log-level","debug","--app-dir","/app"]
