import os, time, json, re, threading
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, Body, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from providers import complete_many, reflect_json
from utils import db

# ✅ Persistent chat memory bridge + saver
from memory import (
    bridge_context as mem_bridge,
    save_chat_turn as mem_save,
    export_state as mem_export,
    set_enabled as mem_set_enabled,
)

# =====================================================
#  DoubleHelix API — Emergent Reflection Engine (v0.9.0)
#  - Non-destructive Clean Eyes (policy-only reset)
#  - Reality Bridge (drift/surprise anchoring)
#  - Illusion Loop (sleep → dream → recall) in tick()
#  - Goal-aware prompts, forgiving /goals input
#  - Lightweight history buffer
#  - ✅ Persistent chat memory (Continuity Bridge + saver)
# =====================================================

load_dotenv()
app = FastAPI(title="DoubleHelix API", version="0.9.0")
db.init()

# -----------------------------------------------------
#  CORS (Env + Fallback)
# -----------------------------------------------------
_origins_env = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
if not _origins_env:
    _origins_env = [
        "http://localhost:5173",
        "https://doublehelix-front.onrender.com",
        "https://doublehelix-frount.onrender.com",  # legacy typo retained for safety
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_env,
    allow_origin_regex=os.getenv("CORS_ALLOW_REGEX", r"^https://.*\.onrender\.com$"),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
#  Bootstrap Defaults
# -----------------------------------------------------
def bootstrap_defaults():
    db.upsert_fact("system.name", "DoubleHelix", 0.95)
    db.upsert_fact("system.version", "0.9.0", 0.9)
    db.upsert_fact("last_boot", time.strftime("%Y-%m-%d %H:%M:%S"), 0.7)

    if not db.kv_get("meta.births"):
        db.kv_upsert("meta.births", {"value": "0"})

    vec = db.get_policy_vector()
    db.set_policy_vector(vec)

    if not db.goals_list():
        db.goals_add("Summarize my current operational principles.")

    if not db.kv_get("thresholds"):
        db.kv_upsert("thresholds", {
            "surprise": float(os.getenv("SURPRISE_THRESHOLD", "0.6")),
            "consolidation_interval": int(os.getenv("CONSOLIDATION_INTERVAL", "12")),
            "meta_interval": int(os.getenv("META_INTERVAL", "48")),
            "memory_max": int(os.getenv("MEMORY_MAX", "1000")),
            "pattern_threshold": int(os.getenv("PATTERN_THRESHOLD", "3")),
            # New knobs:
            "clean_eyes_alpha": float(os.getenv("CLEAN_EYES_ALPHA", "0.5")),  # blend strength
            "clean_eyes_var": float(os.getenv("CLEAN_EYES_VAR", "0.02")),     # variance trigger
            "bridge_surprise": float(os.getenv("BRIDGE_SURPRISE", "0.65")),   # surprise trigger
            "bridge_var": float(os.getenv("BRIDGE_VAR", "0.03")),             # variance trigger
        })

    if not db.kv_get("metrics"):
        db.kv_upsert("metrics", {
            "goals_completed": 0,
            "drafts_per_day": 0,
            "avg_reply_score": 0.0,
            "total_replies": 0
        })

bootstrap_defaults()

REQUESTS_PER_MIN = int(os.getenv("REQUESTS_PER_MIN", "20"))
TOKENS_PER_MIN   = int(os.getenv("TOKENS_PER_MIN", "12000"))
N_SAMPLES        = int(os.getenv("N_SAMPLES", "3"))
DRY_RUN          = os.getenv("DRY_RUN", "true").lower() == "true"

# -----------------------------------------------------
#  Rate Limiting
# -----------------------------------------------------
_window_start = time.time()
_req_count = 0
_token_count = 0

def _maybe_rate_limit(tokens_estimate: int = 0):
    global _window_start, _req_count, _token_count
    now = time.time()
    if now - _window_start >= 60:
        _window_start = now
        _req_count = 0
        _token_count = 0
    if _req_count + 1 > REQUESTS_PER_MIN:
        raise HTTPException(429, "Request rate limit reached")
    if _token_count + tokens_estimate > TOKENS_PER_MIN:
        raise HTTPException(429, "Token rate limit reached")
    _req_count += 1
    _token_count += tokens_estimate

# -----------------------------------------------------
#  History buffer (ring) via KV
# -----------------------------------------------------
HIST_KEY = "history.buffer"
HIST_MAX = int(os.getenv("HISTORY_MAX", "200"))

def _history_get() -> List[Dict[str, Any]]:
    buf = db.kv_get(HIST_KEY) or {"items": []}
    items = buf.get("items", [])
    if not isinstance(items, list):
        items = []
    return items

def _history_append(event: Dict[str, Any]):
    items = _history_get()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    event = {"ts": ts, **event}
    items.append(event)
    if len(items) > HIST_MAX:
        items = items[-HIST_MAX:]
    db.kv_upsert(HIST_KEY, {"items": items})

# -----------------------------------------------------
#  Context + Style
# -----------------------------------------------------
def _context_from_memory(prompt: str, k: int = 5) -> str:
    relevant = db.search_facts(prompt, limit=k) or db.top_facts(limit=k)
    lines = [f"- {r['key']}: {r['value']}" for r in relevant]
    return ("Context facts:\n" + "\n".join(lines) + "\n\n") if lines else ""

def get_style_prompt(vec: Dict[str, float]) -> str:
    evolved = db.kv_get("prompts.style_hint")
    if evolved and evolved.get("text"):
        return evolved["text"]
    return _style_hint(vec)

def _style_hint(vec: Dict[str, float]) -> str:
    c, con, plan, sk = [vec.get(k,0.0) for k in ("creativity","conciseness","planning_focus","skepticism")]
    prefs = []
    if c > 0.1:  prefs.append("emphasize novelty and examples")
    if con > 0.1: prefs.append("be concise and avoid fluff")
    if plan > 0.1:prefs.append("use step-by-step plans where helpful")
    if sk > 0.1:  prefs.append("add caveats when uncertain")
    if not prefs: prefs.append("balanced, helpful responses")
    prefs.append('If calculation is required, respond with JSON tool call {"tool":"calculator","tool_input":"EXPR"}.')
    prefs.append('If memory helps, respond with {"tool":"memory_search","tool_input":"query"}.')
    return "Style preferences: " + "; ".join(prefs) + "."

# -----------------------------------------------------
#  Health
# -----------------------------------------------------
@app.get("/")
@app.get("/health")
@app.get("/healthz")
def health():
    return {"status": "ok"}

# -----------------------------------------------------
#  Scoring + Policy
# -----------------------------------------------------
def _subscores(text: str, memory_text: str) -> Dict[str, float]:
    def bag(s): return set(w.lower() for w in re.findall(r"[a-zA-Z]+", s))
    overlap = len(bag(text) & bag(memory_text)) / (len(bag(text)) + 1e-6)
    novelty = 1.0 - min(overlap, 1.0)
    length = len(text)
    conciseness = max(0.0, 1.0 - (length/600.0))
    planning = 1.0 if re.search(r"\b(1\.|-|\*)", text) or ("\n" in text and len(text.splitlines())>=3) else 0.0
    skepticism = min(1.0, len(re.findall(r"\b(might|may|could|uncertain|not sure)\b", text.lower()))/3.0)
    return {"creativity": novelty, "conciseness": conciseness, "planning_focus": planning, "skepticism": skepticism}

def _score_with_policy(text: str, memory: List[Dict[str,Any]], vec: Dict[str,float]) -> Tuple[float,Dict[str,float]]:
    mem_text = " ".join([m["value"] for m in memory])
    subs = _subscores(text, mem_text)
    score = sum((1.0 + float(vec.get(k,0.0))) * v for k,v in subs.items())
    return round(score,3), subs

def _policy_nudge(vec: Dict[str,float], subs: Dict[str,float], direction: float = 0.06) -> Dict[str,float]:
    new_vec = dict(vec)
    history = db.get_policy_history(5)
    for k,v in subs.items():
        delta = (v - 0.5) * direction
        if abs(delta) < 0.02:
            delta = 0.0
        new_val = new_vec.get(k,0.0) + delta
        if history and len(history) >= 3:
            recent = [h.get(k,0.0) for h in history[-3:] if isinstance(h,dict)]
            new_val = 0.7 * new_val + 0.3 * (sum(recent)/max(1,len(recent)))
        new_vec[k] = max(-1.0, min(1.0, new_val))
    return new_vec

def _policy_variances(window: int = 10) -> Dict[str, float]:
    hist = db.get_policy_history(window)
    variances: Dict[str, float] = {}
    if len(hist) >= 2:
        for trait in ("creativity","conciseness","planning_focus","skepticism"):
            vals = [float(h.get(trait,0.0)) for h in hist if isinstance(h,dict)]
            if len(vals) >= 2:
                m = sum(vals)/len(vals)
                variances[trait] = sum((v-m)**2 for v in vals)/len(vals)
    return {k: round(v,4) for k,v in variances.items()}

# -----------------------------------------------------
#  Memory helpers
# -----------------------------------------------------
def memory_decay(decay: float = 0.01):
    items = db.all_facts()
    for it in items:
        c = max(0.0, float(it.get("confidence", 0.5)) - decay)
        if c > 0.05:
            db.upsert_fact(it["key"], it["value"], c)

def memory_contradictions() -> List[Dict[str, Any]]:
    items = db.all_facts()
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        prefix = it["key"].split(":")[0] if ":" in it["key"] else it["key"]
        buckets.setdefault(prefix, []).append(it)
    conflicts = []
    for prefix, grp in buckets.items():
        unique_vals = {i["value"] for i in grp}
        if len(unique_vals) > 1 and len(grp) >= 2:
            conflicts.append({"prefix": prefix, "items": grp[:3]})
    return conflicts[:5]

def memory_consolidate():
    facts = db.all_facts()
    principle_facts = [f for f in facts if str(f['key']).startswith('principle:')]
    if len(principle_facts) >= 3:
        synthesis_prompt = json.dumps({
            "role": "synthesize",
            "facts": principle_facts[:6],
            "instruction": "Identify common themes across these principles and create a higher-order meta-principle that captures their essence."
        })
        synthesis = reflect_json(synthesis_prompt)
        if isinstance(synthesis, dict) and synthesis.get("principle"):
            births_data = db.kv_get("meta.births") or {"value": "0"}
            births = int(births_data.get("value", "0")) + 1
            db.upsert_fact(f"emergent:{births}", synthesis["principle"], 0.95)
            db.kv_upsert("meta.births", {"value": str(births)})
            db.kv_upsert("illusion.last_meta", {"value": synthesis.get("principle")})

def prune_memory(max_items: int = 1000):
    items = db.all_facts()
    if len(items) > max_items:
        items.sort(key=lambda x: (x.get("confidence", 0.5), x.get("ts", "")))
        for it in items[:len(items)-max_items]:
            db.upsert_fact(it["key"], it["value"], 0.01)

# -----------------------------------------------------
#  Memory command helpers (NEW)
# -----------------------------------------------------
def _memory_status() -> Dict[str, Any]:
    facts = db.all_facts()
    latest = sorted(facts, key=lambda x: x.get("ts",""))[-5:] if facts else []
    return {
        "enabled": True,
        "total_facts": len(facts),
        "latest": [{"key": f["key"], "value": f["value"], "confidence": f.get("confidence", 0.0)} for f in latest]
    }

def _memory_write(text: str, confidence: float = 0.95, key: Optional[str] = None) -> Dict[str, Any]:
    k = key or f"note:{int(time.time())}"
    db.upsert_fact(k, text.strip(), confidence)
    _history_append({"kind": "memory_write", "key": k, "value": text.strip(), "confidence": confidence})
    return {"ok": True, "key": k, "value": text.strip(), "confidence": confidence}

def _memory_recent(limit: int = 5) -> List[Dict[str, Any]]:
    facts = db.all_facts()
    facts = sorted(facts, key=lambda x: x.get("ts",""))[-limit:] if facts else []
    return [{"key": f["key"], "value": f["value"], "confidence": f.get("confidence", 0.0)} for f in facts]

# -----------------------------------------------------
#  Clean Eyes (non-destructive) + Reality Bridge
# -----------------------------------------------------
BASELINE_POLICY_KEY = "cleaneyes.baseline_policy"

def _snapshot_baseline_if_missing():
    if not db.kv_get(BASELINE_POLICY_KEY):
        db.kv_upsert(BASELINE_POLICY_KEY, db.get_policy_vector())

_snapshot_baseline_if_missing()

def _soft_blend(curr: Dict[str,float], base: Dict[str,float], alpha: float) -> Dict[str,float]:
    out = dict(curr)
    for k in ("creativity","conciseness","planning_focus","skepticism"):
        out[k] = float(alpha)*float(base.get(k,0.0)) + float(1.0-alpha)*float(curr.get(k,0.0))
        out[k] = max(-1.0, min(1.0, out[k]))
    return out

def clean_eyes_preview(alpha: Optional[float] = None) -> Dict[str, Any]:
    thresholds = db.kv_get("thresholds") or {}
    alpha = thresholds.get("clean_eyes_alpha", 0.5) if alpha is None else alpha
    curr = db.get_policy_vector()
    base = (db.kv_get(BASELINE_POLICY_KEY) or {})
    proposal = _soft_blend(curr, base, alpha=alpha)
    return {
        "alpha": alpha,
        "current": curr,
        "baseline": base,
        "proposal": proposal,
        "variances": _policy_variances(),
    }

def clean_eyes_apply(alpha: Optional[float] = None) -> Dict[str, Any]:
    preview = clean_eyes_preview(alpha)
    db.set_policy_vector(preview["proposal"])
    _history_append({"kind": "clean_eyes", "alpha": preview["alpha"], "from": preview["current"], "to": preview["proposal"]})
    return preview

def _goal_text() -> str:
    g = db.goal_active()
    if isinstance(g, dict):
        return g.get("text", "") or ""
    return str(g or "")

def _bridge_context(k_goal: int = 1, k_facts: int = 7, k_emergent: int = 2) -> str:
    goal_text = _goal_text()
    facts = db.top_facts(limit=20)
    facts_sorted = sorted(facts, key=lambda x: float(x.get("confidence", 0.5)), reverse=True)
    picked = facts_sorted[:k_facts]
    emergent = [f for f in facts_sorted if str(f.get("key","")).startswith("emergent:")][:k_emergent]
    lines = []
    if goal_text:
        lines.append(f"Active goal: {goal_text}")
    if picked:
        lines.append("Stable anchors:")
        for f in picked:
            lines.append(f"  - {f['key']}: {f['value']}")
    if emergent:
        lines.append("Recent emergent principles:")
        for e in emergent:
            lines.append(f"  - {e['key']}: {e['value']}")
    return ("\n".join(lines) + "\n\n") if lines else ""

def _should_bridge() -> bool:
    thresholds = db.kv_get("thresholds") or {}
    bridge_s = float(thresholds.get("bridge_surprise", 0.65))
    bridge_v = float(thresholds.get("bridge_var", 0.03))
    last_surprise = float((db.kv_get("last.surprise") or {}).get("value", 0.0))
    variances = _policy_variances()
    max_var = max(variances.values()) if variances else 0.0
    return last_surprise >= bridge_s or max_var >= bridge_v

# -----------------------------------------------------
#  Illusion Loop (sleep → dream → recall) within tick()
# -----------------------------------------------------
def illusion_sleep_dream_recall():
    anchors = _bridge_context(k_goal=1, k_facts=7, k_emergent=2)
    dream_req = {
        "phase": "dream",
        "instruction": "Based on the anchors below, speculate a concise, non-redundant synthesis (1-2 sentences) that could guide future replies.",
        "anchors": anchors
    }
    dream = reflect_json(json.dumps(dream_req))
    if isinstance(dream, dict) and dream.get("text"):
        db.kv_upsert("illusion.last_dream", {"value": dream["text"]})

    recall_req = {
        "phase": "recall",
        "instruction": "Distill the most concrete, non-obvious, helpful single principle derived from the dream, in plain language.",
        "dream": dream
    }
    recall = reflect_json(json.dumps(recall_req))
    if isinstance(recall, dict) and recall.get("principle"):
        births_data = db.kv_get("meta.births") or {"value": "0"}
        births = int(births_data.get("value","0")) + 1
        db.upsert_fact(f"emergent:{births}", recall["principle"], 0.95)
        db.kv_upsert("meta.births", {"value": str(births)})
        db.kv_upsert("illusion.last_recall", {"value": recall["principle"]})
        _history_append({"kind": "illusion", "dream": dream, "recall": recall, "birth": births})

# -----------------------------------------------------
#  Perspectives
# -----------------------------------------------------
def perturb_policy(vec: Dict[str, float], profile: str) -> Dict[str, float]:
    p = dict(vec)
    if profile == "explorer":
        p["creativity"] = min(1.0, p.get("creativity",0)+0.3)
        p["skepticism"] = max(-1.0, p.get("skepticism",0)-0.2)
    elif profile == "skeptic":
        p["skepticism"] = min(1.0, p.get("skepticism",0)+0.3)
        p["creativity"] = max(-1.0, p.get("creativity",0)-0.2)
    elif profile == "planner":
        p["planning_focus"] = min(1.0, p.get("planning_focus",0)+0.3)
        p["conciseness"] = min(1.0, p.get("conciseness",0)+0.2)
    return p

# -----------------------------------------------------
#  Tool layer
# -----------------------------------------------------
def _maybe_tool_call(text: str):
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj and "tool_input" in obj:
            return True, obj
    except Exception:
        pass
    return False, {}

def _run_tool(obj: Dict[str, Any]) -> str:
    tool, arg = obj.get("tool"), str(obj.get("tool_input",""))
    if tool == "calculator":
        try:
            if re.fullmatch(r"[0-9\s+\-*/().]+", arg):
                return str(eval(arg, {"__builtins__":{}}))
        except Exception as e:
            return f"calc_error: {e}"
        return "calc_error: invalid input"
    if tool == "memory_search":
        return json.dumps({"memory_hits": db.search_facts(arg, limit=5)})
    if tool == "web_search":
        return "web_search not available in DRY_RUN"
    return f"unknown_tool: {tool}"

# -----------------------------------------------------
#  Chat
# -----------------------------------------------------
class ChatPayload(BaseModel):
    prompt: str
    model: Optional[str] = None
    use_perspectives: bool = False

@app.post("/chat")
def chat(p: ChatPayload):
    try:
        _maybe_rate_limit(max(50, len(p.prompt)//3))

        # ----- Command pre-handler (bypass model) -----
        cmd = p.prompt.strip()

        # /memory status
        if re.fullmatch(r"/memory\s+status", cmd, flags=re.I):
            status = _memory_status()
            return {"reply": json.dumps(status), "meta": {"handled": "memory_status"}}

        # store fact: "..."  OR  write memory: "..."
        m = re.match(r'^(store\s+fact|write\s+memory)\s*:\s*"?(.+?)"?\s*$', cmd, flags=re.I)
        if m:
            text = m.group(2)
            result = _memory_write(text, confidence=0.95)
            return {"reply": f"[memory_ok] key={result['key']}", "meta": {"handled": "memory_write", "value": text}}

        # recall last N facts
        m = re.match(r'^recall\s+last\s+(\d+)\s+facts$', cmd, flags=re.I)
        if m:
            n = max(1, min(100, int(m.group(1))))
            facts = _memory_recent(n)
            return {"reply": json.dumps({"recent": facts}), "meta": {"handled": "memory_recent", "n": n}}

        # ----- Normal model path -----
        vec = db.get_policy_vector()
        style = get_style_prompt(vec)
        memory = db.top_facts(5)

        # Goal-aware + Reality Bridge context
        goal_text = _goal_text()
        goal_prefix = f"Active goal: {goal_text}\n\n" if goal_text else ""
        bridge = _bridge_context(k_goal=1, k_facts=7, k_emergent=2) if _should_bridge() else ""

        # ✅ Persistent Continuity Bridge (recent chats + emergent)
        continuity = mem_bridge(p.prompt)

        # Final context (order matters: goal → runtime bridge → persistent bridge → fact snippets)
        context = goal_prefix + bridge + (continuity or "") + _context_from_memory(p.prompt, k=5)

        outs = []
        used_profiles = []
        if p.use_perspectives:
            for profile in ["base", "explorer", "skeptic", "planner"]:
                p_vec = perturb_policy(vec, profile) if profile != "base" else vec
                p_style = get_style_prompt(p_vec)
                outs.extend(complete_many(context + p.prompt, n=1, model=p.model or "gpt-4o-mini", style_hint=p_style))
                used_profiles.append(profile)
        else:
            outs = complete_many(context + p.prompt, n=N_SAMPLES, model=p.model or "gpt-4o-mini", style_hint=style)
            used_profiles.append("base")

        processed = []
        for o in outs:
            text = o.get("content","")
            is_tool, obj = _maybe_tool_call(text)
            if is_tool:
                text = f"(Tool {obj['tool']} -> {_run_tool(obj)})"
            processed.append(text)

        if not processed:
            processed = ["[no_output]"]

        scored = []
        for t in processed:
            s, subs = _score_with_policy(t, memory, vec)
            scored.append((s, t, subs))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_text, best_subs = scored[0]

        # policy nudge & persist
        vec = _policy_nudge(vec, best_subs, 0.06)
        db.set_policy_vector(vec)

        # metrics + surprise tracking
        metrics = db.kv_get("metrics") or {}
        prev_total = int(metrics.get("total_replies", 0))
        new_total  = prev_total + 1
        prev_avg   = float(metrics.get("avg_reply_score", 0.0))
        metrics["total_replies"]  = new_total
        metrics["avg_reply_score"] = round((prev_avg * prev_total + best_score) / new_total, 3)
        db.kv_upsert("metrics", metrics)

        # Surprise proxy = novelty/creativity subscore
        surprise = float(best_subs.get("creativity", 0.0))
        db.kv_upsert("last.surprise", {"value": surprise})

        # history log
        _history_append({
            "kind": "chat",
            "prompt": p.prompt,
            "active_goal": goal_text,
            "used_perspectives": p.use_perspectives,
            "profiles": used_profiles,
            "candidates": [t for _, t, _ in scored],
            "chosen": best_text,
            "score": best_score,
            "bridge_used": bool(bridge or continuity),
            "surprise": surprise
        })

        # ✅ Persist this chat turn to memory (non-blocking / best-effort)
        try:
            mem_save(
                p.prompt,
                best_text,
                meta={
                    "policy": vec,
                    "score": best_score,
                    "surprise": surprise,
                    "profiles": used_profiles
                }
            )
        except Exception:
            pass

        return {
            "reply": best_text,
            "meta": {
                "score": best_score,
                "surprise": round(surprise, 3),
                "candidates": len(scored),
                "used_perspectives": bool(p.use_perspectives),
                "profiles": used_profiles,
                "policy": vec,
                "bridge_used": bool(bridge or continuity),
            }
        }

    except Exception as e:
        return {"reply": f"[server_error] {type(e).__name__}: {e}", "meta": {"error": True}}

# -----------------------------------------------------
#  Planner / Tick
# -----------------------------------------------------
@app.post("/tick")
def tick():
    ctr = (db.kv_get("tick.counter") or {"n": 0})["n"] + 1
    db.kv_upsert("tick.counter", {"n": ctr})
    thresholds = db.kv_get("thresholds") or {}
    consolidation_interval = int(thresholds.get("consolidation_interval", 12))
    meta_interval = int(thresholds.get("meta_interval", 48))
    actions = []

    # Periodic memory maintenance
    if ctr % consolidation_interval == 0:
        memory_decay(0.02)
        prune_memory(int(thresholds.get("memory_max", 1000)))
        actions.append("memory_decay_and_prune")
        conflicts = memory_contradictions()
        if conflicts:
            db.goals_add(f"Reconcile conflicting facts in '{conflicts[0]['prefix']}'")
            actions.append("conflict_detected")

    # Illusion loop + meta consolidation
    if ctr % meta_interval == 0:
        memory_consolidate()
        illusion_sleep_dream_recall()
        actions.append("meta_reflection_and_illusion")

    # Drift-based maintenance
    variances = _policy_variances()
    max_var = max(variances.values()) if variances else 0.0
    if max_var >= float(thresholds.get("clean_eyes_var", 0.02)):
        clean_eyes_apply(alpha=thresholds.get("clean_eyes_alpha", 0.5))
        actions.append("clean_eyes_applied")

    if _should_bridge():
        db.kv_upsert("bridge.last", {"value": _bridge_context(k_goal=1, k_facts=7, k_emergent=2)})
        actions.append("bridge_refreshed")

    _history_append({"kind": "tick", "n": ctr, "actions": actions, "variances": variances})
    return {"ok": True, "tick": ctr, "actions": actions}

# -----------------------------------------------------
#  Feedback / Emergence
# -----------------------------------------------------
class FeedbackPayload(BaseModel):
    signal: str

@app.post("/feedback")
def feedback(p: FeedbackPayload):
    if p.signal not in ("up","down"):
        raise HTTPException(400,"signal must be 'up' or 'down'")
    total = (db.kv_get("reward.total") or {}).get("total",0)
    total += 1 if p.signal == "up" else -1
    db.kv_upsert("reward.total", {"total": total})
    vec = db.get_policy_vector()
    if p.signal == "up":
        vec["creativity"] = min(1.0, vec.get("creativity",0) + 0.05)
    else:
        vec["conciseness"] = min(1.0, vec.get("conciseness",0) + 0.05)
    db.set_policy_vector(vec)
    _history_append({"kind": "feedback", "signal": p.signal, "reward.total": total, "policy": vec})
    return {"ok": True, "reward.total": total, "policy": vec}

@app.get("/emergence")
def emergence_status():
    births_data = db.kv_get("meta.births") or {"value": "0"}
    births = int(births_data.get("value", "0"))
    metrics = db.kv_get("metrics") or {}
    policy_hist = db.get_policy_history(50)
    variances: Dict[str, float] = {}
    if len(policy_hist) >= 10:
        for trait in ["creativity", "conciseness", "planning_focus", "skepticism"]:
            vals = [h.get(trait, 0.0) for h in policy_hist[-10:] if isinstance(h, dict)]
            if len(vals) >= 2:
                mean = sum(vals)/len(vals)
                var = sum((v - mean)**2 for v in vals)/len(vals)
                variances[trait] = round(var,4)

    last_surprise = (db.kv_get("last.surprise") or {}).get("value", 0.0)
    illusion = {
        "dream": (db.kv_get("illusion.last_dream") or {}).get("value"),
        "recall": (db.kv_get("illusion.last_recall") or {}).get("value"),
    }

    return {
        "emergent_principles": births,
        "policy_stability": variances,
        "avg_reply_score": metrics.get("avg_reply_score", 0),
        "total_interactions": metrics.get("total_replies", 0),
        "current_policy": db.get_policy_vector(),
        "surprise": last_surprise,
        "illusion": illusion,
    }

# -----------------------------------------------------
#  Policy
# -----------------------------------------------------
class PolicyVector(BaseModel):
    creativity: float = 0.0
    conciseness: float = 0.0
    skepticism: float = 0.0
    planning_focus: float = 0.0

@app.get("/policy")
def api_get_policy():
    return JSONResponse(db.get_policy_vector())

@app.post("/policy")
def api_set_policy(vec: PolicyVector):
    db.set_policy_vector(vec.model_dump())
    return {"status": "ok", "policy": vec.model_dump()}

@app.get("/policy/history")
def api_policy_history(limit: int = 20):
    return {"history": db.get_policy_history(last=limit)}

# -----------------------------------------------------
#  Clean Eyes endpoints
# -----------------------------------------------------
@app.get("/clean/preview")
def api_clean_preview(alpha: Optional[float] = None):
    return clean_eyes_preview(alpha)

@app.post("/clean/apply")
def api_clean_apply(alpha: Optional[float] = None):
    return clean_eyes_apply(alpha)

# -----------------------------------------------------
#  Memory endpoints (lightweight)  (NEW)
# -----------------------------------------------------
@app.get("/memory/status")
def api_memory_status():
    return _memory_status()

class MemoryIn(BaseModel):
    text: str
    confidence: Optional[float] = 0.95
    key: Optional[str] = None

@app.post("/memory/fact")
def api_memory_fact(p: MemoryIn):
    return _memory_write(p.text, confidence=float(p.confidence or 0.95), key=p.key)

@app.get("/memory/recent")
def api_memory_recent(limit: int = 5):
    return {"recent": _memory_recent(max(1, min(100, limit)))}

# -----------------------------------------------------
#  History endpoint
# -----------------------------------------------------
@app.get("/history")
def api_history(limit: int = 50):
    items = _history_get()
    return {"history": items[-limit:] if limit and limit > 0 else items}

# -----------------------------------------------------
#  Memory admin endpoints (optional)
# -----------------------------------------------------
@app.get("/memory/export")
def api_memory_export():
    return JSONResponse(mem_export())

@app.post("/memory/enable")
def api_memory_enable(flag: bool = Query(True, description="Enable or disable persistent chat memory")):
    mem_set_enabled(flag)
    return {"ok": True, "enabled": flag}

# -----------------------------------------------------
#  Goals (robust parsing)
# -----------------------------------------------------
class GoalIn(BaseModel):
    text: str

@app.get("/goals")
def api_goals_list():
    return {"goals": db.goals_list()}

@app.post("/goals")
async def api_goals_add(
    request: Request,
    payload: Optional[GoalIn] = Body(None),
    text_qs: Optional[str] = Query(None, alias="text"),
):
    text: Optional[str] = None

    # 1) JSON pydantic
    if payload and getattr(payload, "text", None):
        text = payload.text

    # 1b) raw JSON (any common key)
    if not text:
        try:
            js = await request.json()
            if isinstance(js, dict):
                for k in ("text","goal","value","message"):
                    if k in js and str(js[k]).strip():
                        text = str(js[k]).strip()
                        break
        except Exception:
            pass

    # 2) ?text=...
    if not text and text_qs:
        text = text_qs

    # 3) raw body (text/plain)
    if not text:
        try:
            raw = await request.body()
            raw = raw.decode("utf-8").strip()
            if raw and raw != "{}":
                text = raw
        except Exception:
            pass

    # 4) form data fallbacks
    if not text:
        try:
            form = await request.form()
            for k in ("text","goal","value","message"):
                if k in form and str(form[k]).strip():
                    text = str(form[k]).strip()
                    break
        except Exception:
            pass

    if not text or not text.strip():
        raise HTTPException(400, "Missing goal text")

    gid = db.goals_add(text.strip())
    _history_append({"kind":"goal_add","id":gid,"text":text.strip()})
    return {"status": "ok", "id": gid, "goals": db.goals_list()}

@app.post("/goals/{goal_id}/activate")
def api_goals_activate(goal_id: int):
    db.goals_activate(goal_id)
    _history_append({"kind":"goal_activate","id":goal_id,"active":db.goal_active()})
    return {"status": "ok", "active": goal_id}

@app.get("/goals/active")
def api_goal_active():
    return {"active": db.goal_active()}

# -----------------------------------------------------
#  AutoTick Background Thread
# -----------------------------------------------------
_AUTO_TICK_STARTED = False
AUTO_TICK_ENABLED = os.getenv("AUTO_TICK", "true").lower() == "true"
AUTO_TICK_INTERVAL = int(os.getenv("AUTO_TICK_INTERVAL", "300"))

def auto_tick(interval=AUTO_TICK_INTERVAL):
    while True:
        try:
            print("[AutoTick] Running planner tick...")
            tick()
            print("[AutoTick] Done.\n")
        except Exception as e:
            print("[AutoTick ERROR]", e)
        time.sleep(interval)

if AUTO_TICK_ENABLED and not _AUTO_TICK_STARTED:
    _AUTO_TICK_STARTED = True
    threading.Thread(target=auto_tick, daemon=True).start()
    print(f"[AutoTick] started (interval={AUTO_TICK_INTERVAL}s)")
