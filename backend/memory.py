# backend/memory.py
import json, time, hashlib, os
from typing import Dict, Any, List, Optional
from utils import db

# -----------------------------
# Config
# -----------------------------
MEMORY_ENABLED = os.getenv("PERSIST_CHAT", "true").lower() == "true"
CHAT_NS        = os.getenv("CHAT_NAMESPACE", "chat")
MAX_RECENT     = int(os.getenv("MEMORY_RECENT_WINDOW", "12"))      # recent turns kept in KV
MAX_FACTS      = int(os.getenv("MEMORY_MAX_ITEMS", "2000"))        # hard cap via prune
CONF_DEFAULT   = float(os.getenv("MEMORY_CONFIDENCE", "0.7"))

def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _yyyymmdd() -> str:
    return time.strftime("%Y%m%d")

def _fingerprint(*parts: str, size: int = 10) -> str:
    h = hashlib.sha1(("||".join(parts)).encode("utf-8")).hexdigest()
    return h[:size]

# -----------------------------
# Write path
# -----------------------------
def save_chat_turn(prompt: str, reply: str, meta: Optional[Dict[str,Any]] = None) -> None:
    """
    Persist a compact, queryable record of this chat turn using the existing Facts table.
    Key:   chat:YYYYMMDD:<short_hash>
    Value: JSON blob with prompt/response/meta/policy snapshot
    """
    if not MEMORY_ENABLED:
        return

    date = _yyyymmdd()
    policy = (meta or {}).get("policy", {})
    score = (meta or {}).get("score", None)
    surprise = (meta or {}).get("surprise", None)
    profiles = (meta or {}).get("profiles", [])

    record = {
        "ts": _now_ts(),
        "prompt": prompt,
        "reply": reply,
        "score": score,
        "surprise": surprise,
        "policy": policy,
        "profiles": profiles,
    }

    # Stable-ish id: date + prompt/response rough fp
    fid = _fingerprint(prompt[:160], reply[:160])
    key = f"{CHAT_NS}:{date}:{fid}"
    db.upsert_fact(key, json.dumps(record, ensure_ascii=False), CONF_DEFAULT)

    # Update rolling window in KV
    recent = db.kv_get("memory.recent") or {"items": []}
    items: List[Dict[str,Any]] = recent.get("items", [])
    items.append({"key": key, "prompt": prompt[:240], "reply": reply[:240], "ts": record["ts"], "score": score, "surprise": surprise})
    # trim
    if len(items) > MAX_RECENT:
        items = items[-MAX_RECENT:]
    db.kv_upsert("memory.recent", {"items": items, "n": len(items), "ts": _now_ts()})

    # Update coarse stats
    stats = db.kv_get("memory.stats") or {"turns": 0}
    stats["turns"] = int(stats.get("turns", 0)) + 1
    if surprise is not None:
        s_hist = stats.get("surprise_hist", [])
        s_hist.append(float(surprise))
        s_hist = s_hist[-200:]  # keep short
        stats["surprise_hist"] = s_hist
        stats["avg_surprise"] = round(sum(s_hist)/len(s_hist), 3)
    db.kv_upsert("memory.stats", stats)

    # Respect overall memory cap using existing pruning path (optional)
    _maybe_prune()

def _maybe_prune():
    # Defer to existing pruning with a slightly lower conf floor
    all_f = db.all_facts()
    chat_count = sum(1 for f in all_f if str(f["key"]).startswith(f"{CHAT_NS}:"))
    if chat_count > MAX_FACTS:
        # Soft decay: lower confidence on oldest chat facts → next prune will drop them
        # (We avoid schema changes by reusing the same table.)
        for f in sorted((x for x in all_f if str(x["key"]).startswith(f"{CHAT_NS}:")), key=lambda r: r["ts"])[: (chat_count - MAX_FACTS)]:
            try:
                db.upsert_fact(f["key"], f["value"], conf=0.05)
            except Exception:
                pass

# -----------------------------
# Read / Bridge path
# -----------------------------
def bridge_context(prompt: str, limit_principles: int = 3, limit_recent: int = 4) -> str:
    """
    Build a compact, *useful* continuity string that the LLM can condition on.
    Combines:
      - Active goal
      - Recent emergent principles (emergent:*)
      - Recent chat fingerprints (from KV)
    """
    if not MEMORY_ENABLED:
        return ""

    parts: List[str] = []

    # Active goal
    active = db.goal_active()
    if active:
        parts.append(f"Active goal: {active.get('text','').strip()}")

    # Emergent principles (highest confidence first / newest ts)
    emergent = [f for f in db.all_facts() if str(f["key"]).startswith("emergent:")]
    emergent = sorted(emergent, key=lambda r: (r.get("confidence", 0.0), r.get("ts","")), reverse=True)[:limit_principles]
    if emergent:
        parts.append("Recent emergent principles:")
        for e in emergent:
            parts.append(f"• {e['value']}")

    # Recent chat (rolling KV)
    recent = db.kv_get("memory.recent") or {}
    items = recent.get("items", [])[-limit_recent:]
    if items:
        parts.append("Recent chat snapshots:")
        for it in items:
            # very compact; avoid drowning the prompt
            p = it.get("prompt","").replace("\n"," ")[:120]
            r = it.get("reply","").replace("\n"," ")[:140]
            parts.append(f"– Q:{p} | A:{r}")

    if not parts:
        return ""

    return "Continuity Bridge:\n" + "\n".join(parts) + "\n\n"

# -----------------------------
# Export / Admin
# -----------------------------
def export_state() -> Dict[str, Any]:
    stats = db.kv_get("memory.stats") or {}
    recent = db.kv_get("memory.recent") or {}
    return {
        "enabled": MEMORY_ENABLED,
        "stats": stats,
        "recent": recent,
    }

def set_enabled(flag: bool):
    global MEMORY_ENABLED
    MEMORY_ENABLED = bool(flag)
    db.kv_upsert("memory.enabled", {"value": MEMORY_ENABLED, "ts": _now_ts()})
