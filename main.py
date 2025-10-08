
try:
    from api.admin_tools import router as admin_router
    app.include_router(admin_router, prefix="/admin", tags=["admin"])
    print("[admin_tools] mounted at /admin")
except Exception as _e:
    print("[admin_tools] disabled:", _e)

