# backend/memory.py
# Bounded continuity memory for DoubleHelix (ASCII-safe, restart-persistent)

import os
import json
import time
import re
import hashlib
from typing import Dict, Any, List, Optional

from utils import db

# -----------------------------
# Config / env
# -----------------------------
CHAT_NS        = os.getenv("CHAT_NAMESPACE", "chat")
MAX_RECENT     = int(os.getenv("MEMORY_RECENT_WINDOW", "12"))      # recent turns in KV
MAX_FACTS      = int(os.getenv("MEMORY_MAX_ITEMS", "2000"))        # hard cap for chat facts
CONF_DEFAULT   = float(os.getenv("MEMORY_CONFIDENCE", "0.7"))
ENV_ENABLED    = os.getenv("PERSIST_CHAT", "true").strip().lower() in {"1","true","yes","on"}

# KV keys
KV_ENABLED = "memory.enabled"
KV_RECENT  = "memory.recent"
KV_STATS   = "memory.stats"
KV_MARKERS = "memory.markers"
KV_TOPIC   = "thread.topic"
KV_CLOCK   = "memory.clock"   # temporal continuity key

# Reflection/continuity markers
REFLECTION_MARKERS = [
    r"\billusion of continuity\b",
    r"\bsimulate(?:s|d)? continuity\b",
    r"\b(simulated|apparent)\s+continuity\b",
    r"\bcontinuity bridge\b",
    r"\breality bridge\b",
    r"\bpersistent memory\b",
    r"\b(long[- ]?term memory|retain memory)\b",
    r"\bcarry(?:ing)? forward\b",
    r"\brecall(?:ing)? chat turns?\b",
    r"\bemergent principle\b",
    r"\bclean eyes\b",
    r"\bpolicy(?: vector)?\b",
]

# -----------------------------
# Small helpers
# -----------------------------
def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _yyyymmdd() -> str:
    return time.strftime("%Y%m%d")

def _fingerprint(*parts: str, size: int = 10) -> str:
    h = hashlib.sha1(("||".join(parts)).encode("utf-8")).hexdigest()
    return h[:size]

def _asciify(s: str) -> str:
    if not s:
        return ""
    repl = {
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u2022": "*", "\u00b7": "*",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    out = "".join(repl.get(ch, ch) for ch in s)
    return "".join(ch for ch in out if (ch == "\n") or (32 <= ord(ch) <= 126))

def _is_enabled() -> bool:
    kv = db.kv_get(KV_ENABLED)
    if isinstance(kv, dict) and "value" in kv:
        return bool(kv["value"])
    return ENV_ENABLED

def _regex_any(patterns: List[str], text: str) -> List[str]:
    hits = []
    for p in patterns:
        if re.search(p, text, flags=re.I):
            hits.append(p)
    return hits

# -----------------------------
# Temporal awareness layer
# -----------------------------
def update_clock(event: str = "tick") -> Dict[str, Any]:
    """Record a temporal 'heartbeat' and compute Δt since last one."""
    now_ts = _now_ts()
    clock = db.kv_get(KV_CLOCK) or {"history": []}
    hist = list(clock.get("history", []))

    last_ts = hist[-1]["ts"] if hist else None
    delta_sec = None
    if last_ts:
        try:
            t1 = time.mktime(time.strptime(last_ts, "%Y-%m-%d %H:%M:%S"))
            t2 = time.mktime(time.strptime(now_ts, "%Y-%m-%d %H:%M:%S"))
            delta_sec = round(t2 - t1, 2)
        except Exception:
            delta_sec = None

    hist.append({"ts": now_ts, "event": event, "delta_sec": delta_sec})
    hist = hist[-50:]

    db.kv_upsert(KV_CLOCK, {"history": hist, "last_event": event, "last_ts": now_ts, "delta_sec": delta_sec})
    return {"now": now_ts, "delta_sec": delta_sec, "event": event}

def get_clock() -> Dict[str, Any]:
    clock = db.kv_get(KV_CLOCK) or {}
    hist = clock.get("history", [])
    if hist:
        last = hist[-1]
        return {
            "last_event": last.get("event"),
            "last_ts": last.get("ts"),
            "delta_sec": last.get("delta_sec"),
            "history_len": len(hist)
        }
    return {"last_event": None, "delta_sec": None, "history_len": 0}

# -----------------------------
# Public: quick read helpers
# -----------------------------
def recent_turns(n: int = 5) -> List[Dict[str, Any]]:
    r = db.kv_get(KV_RECENT) or {}
    items = list(r.get("items", []))
    return items[-max(1, min(100, n)):]

def export_state() -> Dict[str, Any]:
    return {
        "enabled": _is_enabled(),
        "stats": db.kv_get(KV_STATS) or {},
        "recent": db.kv_get(KV_RECENT) or {},
        "markers": db.kv_get(KV_MARKERS) or {},
        "topic": db.kv_get(KV_TOPIC) or {},
        "clock": get_clock(),  # ← new temporal data
    }

def set_enabled(flag: bool):
    db.kv_upsert(KV_ENABLED, {"value": bool(flag), "ts": _now_ts()})

# -----------------------------
# Write path
# -----------------------------
def save_chat_turn(prompt: str, reply: str, meta: Optional[Dict[str,Any]] = None) -> None:
    if not _is_enabled():
        return

    prompt = (prompt or "").strip()
    reply  = (reply  or "").strip()
    if not prompt or not reply or reply.startswith("[server_error]"):
        return

    # heartbeat
    update_clock(event="chat_turn")

    date     = _yyyymmdd()
    policy   = (meta or {}).get("policy", {})
    score    = (meta or {}).get("score")
    surprise = (meta or {}).get("surprise")
    profiles = (meta or {}).get("profiles", [])
    normalized = (meta or {}).get("normalized")
    intent     = (meta or {}).get("intent")

    record = {
        "ts": _now_ts(),
        "prompt": prompt,
        "reply": reply,
        "score": score,
        "surprise": surprise,
        "policy": policy,
        "profiles": profiles,
        "normalized": normalized,
        "intent": intent,
    }

    fid = _fingerprint(prompt[:160], reply[:160])
    key = f"{CHAT_NS}:{date}:{fid}"
    db.upsert_fact(key, json.dumps(record, ensure_ascii=False), CONF_DEFAULT)

    recent = db.kv_get(KV_RECENT) or {"items": []}
    items: List[Dict[str,Any]] = list(recent.get("items", []))
    items.append({
        "key": key,
        "prompt": prompt[:240],
        "reply": reply[:240],
        "ts": record["ts"],
        "score": score,
        "surprise": surprise
    })
    if len(items) > MAX_RECENT:
        items = items[-MAX_RECENT:]
    db.kv_upsert(KV_RECENT, {"items": items, "n": len(items), "ts": _now_ts()})

    stats = db.kv_get(KV_STATS) or {"turns": 0}
    stats["turns"] = int(stats.get("turns", 0)) + 1
    if surprise is not None:
        hist = list(stats.get("surprise_hist", []))
        hist.append(float(surprise))
        stats["surprise_hist"] = hist[-200:]
        stats["avg_surprise"]  = round(sum(stats["surprise_hist"]) / len(stats["surprise_hist"]), 3)
    db.kv_upsert(KV_STATS, stats)

    _update_markers(reply)
    _maybe_update_topic(reply)
    _maybe_prune()

# -----------------------------
# Marker + Topic tracking
# -----------------------------
def _update_markers(text: str):
    try:
        hits = _regex_any(REFLECTION_MARKERS, text or "")
        if not hits:
            return
        mk = db.kv_get(KV_MARKERS) or {"counts": {}, "last_seen": {}}
        counts: Dict[str, int] = dict(mk.get("counts", {}))
        last_seen: Dict[str, str] = dict(mk.get("last_seen", {}))
        for pat in hits:
            counts[pat] = int(counts.get(pat, 0)) + 1
            last_seen[pat] = _now_ts()
        mk["counts"] = counts
        mk["last_seen"] = last_seen
        mk["totals"] = {
            "continuity": sum(counts.get(p, 0) for p in REFLECTION_MARKERS if "continuity" in p),
            "memory": sum(counts.get(p, 0) for p in REFLECTION_MARKERS if "memory" in p),
            "bridge": sum(counts.get(p, 0) for p in REFLECTION_MARKERS if "bridge" in p),
        }
        db.kv_upsert(KV_MARKERS, mk)
    except Exception:
        pass

def _maybe_update_topic(reply: str):
    if not reply:
        return
    m = re.search(r"(persistent memory|long[- ]?term memory|reality bridge|continuity|policy(?: vector)?|emergent principle|clean eyes|goal[s]?)",
                  reply, re.I)
    if m:
        topic = m.group(1)
        db.kv_upsert(KV_TOPIC, {"value": topic, "ts": _now_ts()})

def _maybe_prune():
    try:
        all_f = db.all_facts()
        chat_keys = [f for f in all_f if str(f.get("key","")).startswith(f"{CHAT_NS}:")]
        if len(chat_keys) <= MAX_FACTS:
            return
        overflow = len(chat_keys) - MAX_FACTS
        for f in sorted(chat_keys, key=lambda r: r.get("ts",""))[:overflow]:
            try:
                db.upsert_fact(f["key"], f["value"], conf=0.05)
            except Exception:
                pass
    except Exception:
        pass

# -----------------------------
# Read / Bridge path
# -----------------------------
def bridge_context(prompt: str, limit_principles: int = 3, limit_recent: int = 4) -> str:
    if not _is_enabled():
        return ""

    parts: List[str] = []

    active = db.goal_active()
    if active and active.get("text"):
        parts.append("Active goal: " + _asciify(active["text"].strip()))

    topic = db.kv_get(KV_TOPIC) or {}
    if topic.get("value"):
        parts.append("Current topic: " + _asciify(str(topic["value"])))

    # Temporal awareness snippet
    clock = get_clock()
    if clock.get("delta_sec") is not None:
        parts.append(f"Time since last memory event: {clock['delta_sec']}s")

    emergent = [f for f in db.all_facts() if str(f.get("key","")).startswith("emergent:")]
    emergent.sort(key=lambda r: (float(r.get("confidence", 0.0)), r.get("ts","")), reverse=True)
    emergent = emergent[:max(0, int(limit_principles))]
    if emergent:
        parts.append("Recent emergent principles:")
        for e in emergent:
            parts.append("- " + _asciify(str(e.get("value",""))))

    dream  = (db.kv_get("illusion.last_dream")  or {}).get("value")
    recall = (db.kv_get("illusion.last_recall") or {}).get("value")
    if dream or recall:
        parts.append("Illusion loop:")
        if dream:  parts.append("- dream: "  + _asciify(str(dream)))
        if recall: parts.append("- recall: " + _asciify(str(recall)))

    stats = db.kv_get(KV_STATS) or {}
    if "avg_surprise" in stats:
        parts.append(f"Avg surprise (recent): {stats.get('avg_surprise')}")

    recent = db.kv_get(KV_RECENT) or {}
    items = list(recent.get("items", []))[-max(0, int(limit_recent)):]
    if items:
        parts.append("Recent chat snapshots:")
        for it in items:
            p = _asciify((it.get("prompt","") or "").replace("\n"," ")[:120])
            r = _asciify((it.get("reply","")  or "").replace("\n"," ")[:140])
            parts.append(f"- Q: {p} | A: {r}")

    if not parts:
        return ""

    bridge = "Continuity Bridge:\n" + "\n".join(parts) + "\n\n"
    return bridge[:1400]
