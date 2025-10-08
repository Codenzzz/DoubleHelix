# backend/api/helix_verify.py
from fastapi import HTTPException

def require_scopes(scopes):
    def _checker():
        # Temporary permissive mock; always passes
        # You can later expand to verify JWT claims or API tokens here.
        return True
    return _checker
