# shim so backend.main can import admin tools from the top-level package
from api.admin_tools import router
__all__ = ["router"]
