# backend/api/github_bridge.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os, time, hashlib, base64, requests
# 🔒 STRICT import — fail if verify isn’t available
from backend.api.helix_verify import require_scopes

# --- Auth: require "admin.github" scope ---
try:
    # prefer local-style import
    from api.helix_verify import require_scopes
except Exception:
    try:
        # fallback if package name used elsewhere
        from backend.api.helix_verify import require_scopes  # type: ignore
    except Exception:
        # final fallback = permissive (NOT secure); use only during bootstrapping
        def require_scopes(_):
            def _checker(): return True
            return _checker

router = APIRouter(prefix="/admin/github", tags=["admin:github"])

# ----- Models -----
class EditReq(BaseModel):
    path: str                  # repo-relative path, e.g. "api/bridge_test2.txt" or "backend/main.py"
    content: str               # full file contents
    message: str = "Bridge commit"
    branch: str | None = None  # defaults to env GITHUB_BRANCH or "main"
    push: bool = True          # set false to just write locally (no commit)

# ----- Helpers -----
def _safe_local_path(rel_path: str) -> Path:
    """Ensure we only write inside the current working directory."""
    root = Path(os.getcwd()).resolve()
    p = (root / rel_path).resolve()
    if not str(p).startswith(str(root)):
        raise HTTPException(400, "Invalid path")
    return p

def _gh_cfg():
    repo = os.getenv("GITHUB_REPO")  # "owner/repo"
    token = os.getenv("GITHUB_TOKEN")
    user = os.getenv("GITHUB_USER") or (repo.split("/", 1)[0] if repo and "/" in repo else None)
    branch = os.getenv("GITHUB_BRANCH", "main")
    return user, repo, token, branch

def _gh_headers(token: str):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "HelixBridge",
    }

def _gh_get_file_sha(repo: str, path: str, branch: str, token: str) -> str | None:
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r = requests.get(url, params={"ref": branch}, headers=_gh_headers(token), timeout=20)
    if r.status_code == 200:
        return r.json().get("sha")
    if r.status_code == 404:
        return None
    raise HTTPException(r.status_code, f"GitHub probe failed: {r.text}")

def _gh_upsert_file(repo: str, path: str, branch: str, message: str, content_str: str, token: str):
    # base64-encode content
    b64 = base64.b64encode(content_str.encode("utf-8")).decode("ascii")
    sha = _gh_get_file_sha(repo, path, branch, token)
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    payload = {
        "message": message,
        "content": b64,
        "branch": branch,
        **({"sha": sha} if sha else {}),
    }
    r = requests.put(url, json=payload, headers=_gh_headers(token), timeout=30)
    if r.status_code not in (200, 201):
        raise HTTPException(r.status_code, f"GitHub commit failed: {r.text}")
    j = r.json()
    commit = (j.get("commit") or {})
    html_url = commit.get("html_url")
    return {
        "committed": True,
        "path": path,
        "branch": branch,
        "sha": (j.get("content") or {}).get("sha"),
        "commit_url": html_url,
    }

# ----- Routes (SECURED) -----
@router.get("/ping", dependencies=[Depends(require_scopes(["admin.github"]))])
def ping():
    user, repo, token, branch = _gh_cfg()
    return {
        "ok": True,
        "module": "github_bridge",
        "ts": time.time(),
        "env": {"user": bool(user), "repo": bool(repo), "token": bool(token), "branch": branch},
    }

@router.post("/edit", dependencies=[Depends(require_scopes(["admin.github"]))])
def edit_file(req: EditReq):
    # 1) Always write locally (predictable local dev)
    target = _safe_local_path(req.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    sha_local = hashlib.sha256(req.content.encode("utf-8")).hexdigest()

    # 2) Optionally push to GitHub
    if req.push:
        user, repo, token, default_branch = _gh_cfg()
        branch = req.branch or default_branch or "main"
        if not (repo and token):
            raise HTTPException(400, "Missing GITHUB_TOKEN or GITHUB_REPO")
        gh = _gh_upsert_file(
            repo=repo, path=req.path, branch=branch,
            message=req.message, content_str=req.content, token=token
        )
        return {"ok": True, "mode": "local+github", "path": str(target), "sha256": sha_local, **gh}

    # local-only
    return {"ok": True, "mode": "local-only", "path": str(target), "sha256": sha_local}
