# backend/api/self_update.py
from fastapi import APIRouter, Body, HTTPException
from backend.api.github_bridge import EditReq, edit_file

router = APIRouter(prefix="/admin/self", tags=["admin:self"])

@router.post("/update")
def self_update(
    path: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
    message: str = Body("auto-update", embed=True),
):
    if not path or content is None:
        raise HTTPException(400, "path and content are required")
    req = EditReq(path=path, content=content, message=message)
    # Reuse the existing bridge logic (writes locally + pushes to Git if env is set)
    return edit_file(req)
