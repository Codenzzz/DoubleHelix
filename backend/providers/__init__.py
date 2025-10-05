
import os, time
from typing import Dict, List
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
DEFAULT_MODEL = "gpt-4o-mini"

def complete_many(prompt: str, n: int, model: str | None = None, style_hint: str = "", **kwargs) -> List[Dict]:
    model = model or DEFAULT_MODEL
    if DRY_RUN:
        styles = ["(succinct)", "(creative)", "(structured)"]
        outs = []
        for i in range(n):
            tag = styles[i % len(styles)]
            outs.append({"provider":"stub","model":model,"created":int(time.time()),
                         "content": f"[DRY_RUN] {tag} {style_hint} → {prompt[:200]}"})
        return outs
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        from .openai_provider import openai_chat_many
        return openai_chat_many(prompt, n=n, model=model, system_hint=style_hint, **kwargs)
    raise RuntimeError(f"Unsupported provider: {provider}")

def reflect_json(prompt: str, model: str | None = None) -> Dict:
    model = model or DEFAULT_MODEL
    if DRY_RUN:
        return {"best_candidate_index": 0, "new_fact": "Concise answers preferred in this context.", "policy_adjustment": {"conciseness": 0.05}}
    provider = os.getenv("LLM_PROVIDER", "openai")
    if provider == "openai":
        from .openai_provider import openai_reflect_json
        return openai_reflect_json(prompt, model=model)
    raise RuntimeError(f"Unsupported provider: {provider}")
