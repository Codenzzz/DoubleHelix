# backend/api/ops_ui.py
from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from backend.api.helix_verify import _decode_bearer
from backend.api.github_bridge import EditReq, edit_file

router = APIRouter(prefix="/ops", tags=["ops"])

HTML = """<!doctype html>
<meta charset="utf-8" />
<title>Helix Ops</title>
<style>
  :root { color-scheme: light dark; }
  body{font-family:system-ui,Segoe UI,Arial;margin:2rem;max-width:920px}
  form{margin:1.25rem 0;padding:1rem;border:1px solid #ddd;border-radius:12px}
  input,textarea{width:100%;padding:.55rem;margin:.25rem 0;font-family:ui-monospace,Consolas,monospace}
  button{padding:.6rem 1rem;border:0;border-radius:10px;background:#111;color:#fff;cursor:pointer}
  small{color:#666} label{font-weight:600}
  pre{white-space:pre-wrap;word-wrap:anywhere;background:#0001;padding:1rem;border-radius:10px}
</style>

<h1>Helix Ops</h1>
<p><small>The page is public, but actions require a JWT with <code>admin.github</code>.
Paste your token below for each action.</small></p>

<h2>Local write (no Git push)</h2>
<form method="post" action="/ops/local">
  <label>JWT token (admin.github)</label>
  <input name="token" placeholder="eyJhbGciOi...">
  <label>Path (repo-relative)</label>
  <input name="path" placeholder="backend/api/new_module.py" required>
  <label>Message</label>
  <input name="message" value="ops: local edit">
  <label>Content</label>
  <textarea name="content" rows="14" placeholder="# your code here" required></textarea>
  <button>Write locally</button>
</form>

<h2>Write + GitHub push</h2>
<form method="post" action="/ops/push">
  <label>JWT token (admin.github)</label>
  <input name="token" placeholder="eyJhbGciOi...">
  <label>Path (repo-relative)</label>
  <input name="path" placeholder="backend/api/new_module.py" required>
  <label>Commit message</label>
  <input name="message" value="ops: push edit">
  <label>Content</label>
  <textarea name="content" rows="14" placeholder="# your code here" required></textarea>
  <button>Write & Push</button>
</form>
"""

def _verify_token_or_403(token: str | None, need_scope: str = "admin.github"):
    if not token:
        raise HTTPException(401, "Missing token")
    claims = _decode_bearer(f"Bearer {token}")
    scopes = set(claims.get("scopes", []))
    if need_scope not in scopes:
        raise HTTPException(403, f"Missing scope: {need_scope}")
    return claims

@router.get("", response_class=HTMLResponse)
def ops_index():
    return HTML

@router.post("/local", response_class=HTMLResponse)
def ops_local(
    token: str | None = Form(None),
    path: str = Form(...),
    message: str = Form("ops: local edit"),
    content: str = Form(...)
):
    _verify_token_or_403(token, "admin.github")
    res = edit_file(EditReq(path=path, content=content, message=message, push=False))
    return HTMLResponse(f"<h3>Local write OK</h3><pre>{res}</pre>")

@router.post("/push", response_class=HTMLResponse)
def ops_push(
    token: str | None = Form(None),
    path: str = Form(...),
    message: str = Form("ops: push edit"),
    content: str = Form(...)
):
    _verify_token_or_403(token, "admin.github")
    res = edit_file(EditReq(path=path, content=content, message=message, push=True))
    return HTMLResponse(f"<h3>Write + Push OK</h3><pre>{res}</pre>")