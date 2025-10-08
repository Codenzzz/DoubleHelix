from fastapi import APIRouter, Body
import time

router = APIRouter(prefix="/tot", tags=["tot"])

@router.get("/ping")
def tot_ping():
    return {"ok": True, "module": "tot", "ts": time.time()}

@router.post("/solve")
def tot_solve(prompt: str = Body(..., embed=True), steps: int = 3):
    """
    Minimal Tree-of-Thought stub: returns skeleton branches only.
    We'll flesh this out after wiring the router into main.py.
    """
    branches = []
    seed = hash(prompt) & 0xffff
    for i in range(max(1, int(steps))):
        branches.append({"id": i+1, "thought": f"step {i+1} for {prompt}", "score": (seed % 97)/100})
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
    return {"ok": True, "prompt": prompt, "branches": branches}