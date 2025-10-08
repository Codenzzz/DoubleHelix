# backend/api/github_bridge.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from api.helix_verify import require_scopes
from pathlib import Path
import subprocess, os, time, hashlib

router = APIRouter(prefix="/admin/github", tags=["admin:github"])

class EditReq(BaseModel):
    path: str
    content: str
    message: str = "bridge commit"

def _safe_path(rel_path: str) -> Path:
    root = Path(os.getcwd()).resolve()
    p = (root / rel_path).resolve()
    if not str(p).startswith(str(root)):
        raise HTTPException(400, "Invalid path")
    return p

@router.get("/ping")
def ping():
    return {"ok": True, "module": "github_bridge", "ts": time.time()}

def _git(*args, cwd: Path):
    """Run a git command and return output or raise error."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"Git command failed: {' '.join(args)}\n{e.stderr}")

@router.post("/edit", dependencies=[Depends(require_scopes(["github.write"]))])
def edit_and_commit(req: EditReq):
    target = _safe_path(req.path)
    repo_root = Path(os.getcwd()).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")

    sha = hashlib.sha256(req.content.encode("utf-8")).hexdigest()
    commit_msg = req.message or f"bridge update: {target.name}"

    # Commit + push logic
    branch = os.getenv("GITHUB_BRANCH", "main")
    try:
        _git("add", str(target.relative_to(repo_root)), cwd=repo_root)
        _git("commit", "-m", commit_msg, cwd=repo_root)
        # Inject token for HTTPS push
        token = os.getenv("GITHUB_TOKEN")
        user = os.getenv("GITHUB_USER")
        repo = os.getenv("GITHUB_REPO")

        if not token or not user or not repo:
            raise HTTPException(500, "Missing GITHUB_TOKEN, GITHUB_USER, or GITHUB_REPO")

        repo_url = f"https://{user}:{token}@github.com/{repo}.git"
        _git("push", repo_url, f"HEAD:{branch}", cwd=repo_root)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Commit/push failed: {e}")

    return {
        "ok": True,
        "path": str(target),
        "sha256": sha,
        "message": commit_msg,
        "branch": branch,
    }
