#!/bin/sh
set -e

echo "=== BOOTCHECK (entrypoint) ==="
python - <<'PY'
import os, sys, importlib, inspect, traceback
print("cwd:", os.getcwd())
print("sys.executable:", sys.executable)
print("PYTHONPATH:", os.getenv("PYTHONPATH"))
print("sys.path[:5]:", sys.path[:5])

for p in ["/app", "/app/backend"]:
    print(f"exists {p}:", os.path.isdir(p))
    if os.path.isdir(p):
        try:
            print(f"ls {p}:", sorted(os.listdir(p))[:40])
        except Exception as e:
            print(f"ls {p} error:", e)

try:
    m = importlib.import_module("backend.main")
    print("IMPORT OK:", inspect.getfile(m), "has app:", hasattr(m, "app"))
except Exception:
    print("IMPORT FAILED: backend.main")
    traceback.print_exc()
    raise
PY
echo "=== /BOOTCHECK ==="

if [ "$#" -gt 0 ]; then
  echo "ENTRYPOINT: execing provided command: $@"
  exec "$@"
else
  echo "ENTRYPOINT: no command provided, starting default uvicorn"
  exec python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level debug --app-dir /app
fi
