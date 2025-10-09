# api/self_update.py
from fastapi import APIRouter, Body, HTTPException
from api.github_bridge import EditReq, edit_file
import os

router = APIRouter(prefix="/admin/self", tags=["admin:self"])

@router.get("/status")
def self_status():
    """Lightweight probe to confirm the router is mounted."""
    return {
        "ok": True,
        "branch": os.getenv("GITHUB_BRANCH", "main"),
        "mode": "local+github" if os.getenv("GITHUB_TOKEN") else "local",
    }

@router.post("/update")
def self_update(
    path: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
    message: str = Body("auto-update", embed=True),
):
    if not path or content is None:
        raise HTTPException(400, "path and content are required")
    req = EditReq(path=path, content=content, message=message)
    # Reuse bridge logic (writes locally + commits/pushes if env set)
    return edit_file(req)


# api/self_update.py
import time
from fastapi import APIRouter, Body, HTTPException
from api.github_bridge import EditReq, edit_file

router = APIRouter(prefix="/admin/self", tags=["admin:self"])

@router.get("/status")
def self_status():
    return {"ok": True, "ts": time.time()}

@router.post("/update")
def self_update(
    path: str = Body(..., embed=True),
    content: str = Body(..., embed=True),
    message: str = Body("auto-update", embed=True),
):
    if not path or content is None:
        raise HTTPException(400, "path and content are required")
    req = EditReq(path=path, content=content, message=message)
    return edit_file(req)
