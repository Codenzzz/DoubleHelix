# providers/openai_provider.py
import os, time, json
from typing import Dict, List, Any, Union
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _client() -> httpx.Client:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return httpx.Client(
        timeout=60,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )

def _normalize_messages(
    prompt_or_messages: Union[str, List[Dict[str, Any]]],
    system_hint: str = ""
) -> List[Dict[str, str]]:
    """
    Accepts a single user prompt (str) OR a ready 'messages' list.
    Ensures everything is a proper {'role','content': str} message.
    """
    msgs: List[Dict[str, str]] = []

    if isinstance(prompt_or_messages, str):
        if system_hint:
            msgs.append({"role": "system", "content": str(system_hint)})
        msgs.append({"role": "user", "content": str(prompt_or_messages)})
        return msgs

    if not isinstance(prompt_or_messages, list) or not prompt_or_messages:
        raise ValueError("messages must be a non-empty list or a string prompt")

    if system_hint:
        msgs.append({"role": "system", "content": str(system_hint)})

    for m in prompt_or_messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"invalid role in messages: {role}")
        # Coerce non-strings to strings to avoid 400s
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        msgs.append({"role": role, "content": content})

    return msgs

# -----------------------------
# Chat completions (multi)
# -----------------------------
def openai_chat_many(
    prompt_or_messages: Union[str, List[Dict[str, Any]]],
    n: int,
    model: str = DEFAULT_MODEL,
    system_hint: str = ""
) -> List[Dict]:
    """
    Calls Chat Completions and returns a list of outputs:
      [{'provider':'openai','model':..., 'created':..., 'content':..., 'raw': choice}, ...]
    """
    url = "https://api.openai.com/v1/chat/completions"
    try:
        messages = _normalize_messages(prompt_or_messages, system_hint=system_hint)
    except Exception as e:
        return [{"content": f"[provider_error] message_normalization: {e}"}]

    # Clamp n (defense in depth)
    try:
        n = max(1, min(8, int(n)))
    except Exception:
        n = 1

    body = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "n": n,
        "temperature": 0.7,
    }

    try:
        with _client() as client:
            r = client.post(url, json=body)
            if r.status_code >= 400:
                # surface OpenAI's error text rather than a generic 400
                try:
                    err = r.json()
                    msg = err.get("error", {}).get("message", r.text)
                except Exception:
                    msg = r.text
                return [{"content": f"[provider_error] OpenAI {r.status_code}: {msg}"}]

            data = r.json()

        outs = []
        for choice in data.get("choices", []):
            text = (choice.get("message", {}) or {}).get("content", "")
            outs.append({
                "provider": "openai",
                "model": model or DEFAULT_MODEL,
                "created": int(time.time()),
                "content": (text or "").strip(),
                "raw": choice,
            })
        if not outs:
            return [{"content": "[provider_error] empty choices"}]
        return outs

    except Exception as e:
        return [{"content": f"[provider_error] {type(e).__name__}: {e}"}]

# -----------------------------
# Reflection module (used by /chat and /meta_reflect)
# -----------------------------
def openai_reflect_json(prompt: str, model: str = DEFAULT_MODEL) -> Dict:
    url = "https://api.openai.com/v1/chat/completions"
    sys_msg = "You are a reflection module. Reply ONLY with valid JSON that matches the required schema."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": str(prompt)},
    ]
    body = {"model": model or DEFAULT_MODEL, "messages": messages, "temperature": 0.2}

    try:
        with _client() as client:
            r = client.post(url, json=body)
            if r.status_code >= 400:
                try:
                    err = r.json()
                    msg = err.get("error", {}).get("message", r.text)
                except Exception:
                    msg = r.text
                return {"note": "reflection_provider_error", "error": f"OpenAI {r.status_code}: {msg}"}

            data = r.json()

        text = data["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(text)
        except Exception as parse_e:
            return {"note": "reflection_parse_error", "error": str(parse_e), "raw": text}

    except Exception as e:
        return {"note": "reflection_transport_error", "error": str(e)}
