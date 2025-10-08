from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
import os

router = APIRouter()

SECRET = os.getenv("HELIX_TRIGGER_SECRET")

def _require_token(token: str):
    if not SECRET:
        raise HTTPException(500, "HELIX_TRIGGER_SECRET not configured")
    if token != SECRET:
        raise HTTPException(401, "bad token")

def _safe_path(rel_path: str) -> Path:
    root = Path(".").resolve()
    p = (root / rel_path).resolve()
    if not str(p).startswith(str(root)):
        raise HTTPException(400, "path traversal")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"file not found: {rel_path}")
    return p

@router.get("/files/get")
def files_get(
    path: str = Query(..., description="repo-relative path, e.g. 'api/chat.py' or 'main.py'"),
    token: str = Query(..., description="must equal HELIX_TRIGGER_SECRET"),
    max_bytes: int = Query(200000, ge=1, le=1000000),
):
    _require_token(token)
    target = _safe_path(path)
    data = target.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        text = data.decode("utf-8", errors="replace")
        return {"ok": True, "path": path, "size": target.stat().st_size, "utf8": True, "content": text}
    except Exception:
        # binary fallback (base64 would be heavier; we keep it simple)
        return {"ok": True, "path": path, "size": target.stat().st_size, "utf8": False}
