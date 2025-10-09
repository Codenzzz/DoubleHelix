# backend/api/helix_auth.py
from __future__ import annotations

import os, time, jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET = os.getenv("HELIXBRIDGE_JWT_SECRET", "")
ISS = os.getenv("JWT_ISS", "helix")
AUD = os.getenv("JWT_AUD", "helix-clients")

class TokenReq(BaseModel):
    sub: str = "chatgpt"
    scopes: list[str] = ["memory.read", "memory.write"]

@router.post("/issue")
def issue_token(p: TokenReq):
    if not SECRET:
        raise HTTPException(500, "Missing HELIXBRIDGE_JWT_SECRET")
    now = int(time.time())
    payload = {"iss": ISS, "aud": AUD, "sub": p.sub, "scopes": p.scopes, "iat": now, "exp": now + 3600}
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    return {"token": token, "exp": payload["exp"], "scopes": p.scopes}
