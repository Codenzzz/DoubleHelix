# backend/api/helix_verify.py
from __future__ import annotations

import os
import jwt
from fastapi import HTTPException, Header

# Env config
SECRET = os.getenv("HELIXBRIDGE_JWT_SECRET", "")
ISS = os.getenv("JWT_ISS", "helix")
AUD = os.getenv("JWT_AUD", "helix-clients")
FULL_CLIENT = os.getenv("HELIX_FULL_ACCESS_CLIENT", "chatgpt")
ALLOW_FULL = os.getenv("ALLOW_FULL_ACCESS", "false").lower() == "true"

def _decode_bearer(auth_header: str | None):
    if not SECRET:
        raise HTTPException(500, "Missing HELIXBRIDGE_JWT_SECRET")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = auth_header.split(" ", 1)[1]
    try:
        # Validate signature + issuer + audience
        claims = jwt.decode(
            token,
            SECRET,
            algorithms=["HS256"],
            audience=AUD,
            issuer=ISS,
        )
        return claims
    except jwt.PyJWTError as e:
        raise HTTPException(401, f"Invalid token: {e}")

def require_scopes(required: list[str]):
    """
    FastAPI dependency:
      - Allows full bypass if ALLOW_FULL_ACCESS=true and sub == FULL_CLIENT
      - Otherwise requires all scopes in `required` to be present in JWT `scopes`
    """
    def _dep(authorization: str | None = Header(None)):
        claims = _decode_bearer(authorization)
        sub = claims.get("sub", "")
        scopes = set(claims.get("scopes", []))
        if ALLOW_FULL and sub == FULL_CLIENT:
            return claims  # full-access bypass
        missing = [s for s in required if s not in scopes]
        if missing:
            raise HTTPException(403, f"Missing scopes: {missing}")
        return claims
    return _dep
