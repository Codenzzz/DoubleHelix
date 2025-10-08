
try:

    print("[admin_tools] mounted at /admin")
except Exception as _e:
    print("[admin_tools] disabled:", _e)

# ---- Optional admin file-browser mount (safe dynamic import) ----
try:
    import importlib, pathlib, sys, os
    api_dir = pathlib.Path(__file__).parent / "api"
    # Make sure /app is on sys.path (should be by default, but belt & braces)
    root_dir = str(pathlib.Path(__file__).parent.resolve())
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    # If the module file exists, import and mount
    if (api_dir / "admin_tools.py").exists():
        admin_mod = importlib.import_module("api.admin_tools")
        app.include_router(admin_mod.router, prefix="/admin", tags=["admin"])
        print("[admin_tools] mounted at /admin")
    else:
        print("[admin_tools] not found; skipping")
except Exception as e:
    print("[admin_tools] disabled:", e)

