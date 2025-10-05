import os, time, json
from typing import Dict, List
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def _client():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return httpx.Client(
        timeout=60,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )

# -----------------------------
# Chat completions (multi)
# -----------------------------
def openai_chat_many(prompt: str, n: int, model: str = "gpt-4o-mini", system_hint: str = "") -> List[Dict]:
    url = "https://api.openai.com/v1/chat/completions"
    messages = []
    if system_hint:
        messages.append({"role": "system", "content": system_hint})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "n": n,
        "temperature": 0.7
    }

    try:
        with _client() as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            data = r.json()

        outs = []
        for choice in data.get("choices", []):
            text = choice.get("message", {}).get("content", "").strip()
            outs.append({
                "provider": "openai",
                "model": model,
                "created": int(time.time()),
                "content": text,
                "raw": choice,
            })
        return outs
    except Exception as e:
        return [{"content": f"[provider_error] {e}"}]

# -----------------------------
# Reflection module (used by /chat and /meta_reflect)
# -----------------------------
def openai_reflect_json(prompt: str, model: str = "gpt-4o-mini") -> Dict:
    url = "https://api.openai.com/v1/chat/completions"
    sys_msg = "You are a reflection module. Reply ONLY with valid JSON that matches the required schema."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    body = {"model": model, "messages": messages, "temperature": 0.2}

    try:
        with _client() as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            data = r.json()

        text = data["choices"][0]["message"]["content"].strip()
        return json.loads(text)
    except Exception as e:
        return {"note": "reflection_parse_error", "error": str(e)}
