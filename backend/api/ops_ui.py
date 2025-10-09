# backend/api/ops_ui.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse
from backend.api.helix_verify import require_scopes
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
<p><small>Edits go through your secured bridge. Auth required.</small></p>

<h2>Local write (no Git push)</h2>
<form method="post" action="/ops/local">
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
  <label>Path (repo-relative)</label>
  <input name="path" placeholder="backend/api/new_module.py" required>
  <label>Commit message</label>
  <input name="message" value="ops: push edit">
  <label>Content</label>
  <textarea name="content" rows="14" placeholder="# your code here" required></textarea>
  <button>Write & Push</button>
</form>
"""

@router.get("", response_class=HTMLResponse, dependencies=[Depends(require_scopes(["admin.github"]))])
def ops_index():
    return HTML

@router.post("/local", response_class=HTMLResponse, dependencies=[Depends(require_scopes(["admin.github"]))])
def ops_local(path: str = Form(...), message: str = Form("ops: local edit"), content: str = Form(...)):
    req = EditReq(path=path, content=content, message=message, push=False)
    result = edit_file(req)
    return f"<pre>{result}</pre><p><a href='/ops'>Back</a></p>"

@router.post("/push", response_class=HTMLResponse, dependencies=[Depends(require_scopes(["admin.github"]))])
def ops_push(path: str = Form(...), message: str = Form("ops: push edit"), content: str = Form(...)):
    req = EditReq(path=path, content=content, message=message, push=True)
    result = edit_file(req)
    return f"<pre>{result}</pre><p><a href='/ops'>Back</a></p>"