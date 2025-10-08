from api.admin_tools import router as admin_router
app.include_router(admin_router, prefix="/admin", tags=["admin"])
