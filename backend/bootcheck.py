# backend/bootcheck.py
import os, sys, importlib, inspect, traceback

print("=== BOOTCHECK ===")
print("cwd:", os.getcwd())
print("sys.executable:", sys.executable)
print("PYTHONPATH:", os.getenv("PYTHONPATH"))
print("sys.path[:5]:", sys.path[:5])

for p in ["/app", "/app/backend"]:
    print(f"exists {p}:", os.path.isdir(p))
    try:
        print(f"ls {p}:", os.listdir(p) if os.path.isdir(p) else "N/A")
    except Exception as e:
        print(f"ls {p} error:", e)

try:
    m = importlib.import_module("backend.main")
    print("IMPORT OK:", inspect.getfile(m), "has app:", hasattr(m, "app"))
except Exception:
    print("IMPORT FAILED: backend.main")
    traceback.print_exc()
    raise

print("=== /BOOTCHECK ===")
