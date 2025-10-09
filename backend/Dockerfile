# backend/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Put code somewhere untouched by the Render disk mount
WORKDIR /opt/app

# Install deps first (cache layer)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole repo
COPY . .

# Make sure Python imports from /opt/app first
ENV PYTHONPATH=/opt/app

# Sanity check at build time
RUN python - <<'PY'
import os, importlib, sys
print("build-check: cwd", os.getcwd())
print("build-check: have /opt/app/backend:", os.path.isdir("/opt/app/backend"))
print("build-check: list /opt/app/backend:", os.listdir("/opt/app/backend")[:8])
importlib.import_module("backend.main")
print("build-check: import backend.main OK")
PY

EXPOSE 8000
# Run from /opt/app so top-level "backend" resolves to /opt/app/backend
CMD ["python","-m","uvicorn","backend.main:app","--host","0.0.0.0","--port","8000","--app-dir","/opt/app"]
