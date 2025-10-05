import os, time, json, re, itertools, threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

from providers import complete_many, reflect_json
from utils import db

# =====================================================
#  DoubleHelix API  —  Emergent Reflection Engine
# =====================================================

load_dotenv()
app = FastAPI(title="DoubleHelix API", version="0.8.1")
db.init()

# -----------------------------------------------------
#  CORS (Env + Fallback)
# -----------------------------------------------------
_origins_env = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
if not _origins_env:
    _origins_env = [
        "http://localhost:5173",
        "https://doublehelix-front.onrender.com",
        "https://doublehelix-frount.onrender.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_env,
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
#  Bootstrap Defaults
# -----------------------------------------------------
def bootstrap_defaults():
    db.upsert_fact("system.name", "DoubleHelix", 0.95)
    db.upsert_fact("system.version", "0.8.1", 0.9)
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
#  Context + Style
# -----------------------------------------------------
def _context_from_memory(prompt: str) -> str:
    relevant = db.search_facts(prompt, limit=5) or db.top_facts(limit=5)
    lines = [f"- {r['key']}: {r['value']}" for r in relevant]
    return "Context facts:\n" + "\n".join(lines) + "\n\n" if lines else ""

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
#  Health Check
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
    return {"creativity":novelty,"conciseness":conciseness,"planning_focus":planning,"skepticism":skepticism}

def _score_with_policy(text: str, memory: List[Dict[str,Any]], vec: Dict[str,float]) -> tuple[float,Dict[str,float]]:
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

# -----------------------------------------------------
#  Memory
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

def prune_memory(max_items: int = 1000):
    items = db.all_facts()
    if len(items) > max_items:
        items.sort(key=lambda x: (x.get("confidence", 0.5), x.get("ts", "")))
        for it in items[:len(items)-max_items]:
            db.upsert_fact(it["key"], it["value"], 0.01)

# -----------------------------------------------------
#  Multi-agent perspectives
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
#  Chat Route
# -----------------------------------------------------
class ChatPayload(BaseModel):
    prompt: str
    model: str | None = None
    use_perspectives: bool = False

@app.post("/chat")
def chat(p: ChatPayload):
    try:
        _maybe_rate_limit(max(50, len(p.prompt)//3))

        vec = db.get_policy_vector()
        style = get_style_prompt(vec)
        context = _context_from_memory(p.prompt)
        memory = db.top_facts(5)

        outs = []
        if p.use_perspectives:
            for profile in ["base", "explorer", "skeptic", "planner"]:
                p_vec = perturb_policy(vec, profile) if profile != "base" else vec
                p_style = get_style_prompt(p_vec)
                outs.extend(complete_many(context + p.prompt, n=1, model=p.model or "gpt-4o-mini", style_hint=p_style))
        else:
            outs = complete_many(context + p.prompt, n=N_SAMPLES, model=p.model or "gpt-4o-mini", style_hint=style)

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

        vec = _policy_nudge(vec, best_subs, 0.06)
        db.set_policy_vector(vec)

        return {
            "reply": best_text,
            "meta": {"score": best_score, "candidates": len(scored), "policy": vec}
        }

    except Exception as e:
        return {"reply": f"[server_error] {type(e).__name__}: {e}", "meta": {"error": True}}

# -----------------------------------------------------
#  Planner / Tick Loop
# -----------------------------------------------------
@app.post("/tick")
def tick():
    ctr = (db.kv_get("tick.counter") or {"n": 0})["n"] + 1
    db.kv_upsert("tick.counter", {"n": ctr})
    thresholds = db.kv_get("thresholds") or {}
    consolidation_interval = int(thresholds.get("consolidation_interval", 12))
    meta_interval = int(thresholds.get("meta_interval", 48))
    actions = []

    if ctr % consolidation_interval == 0:
        memory_decay(0.02)
        prune_memory(int(thresholds.get("memory_max", 1000)))
        actions.append("memory_decay_and_prune")
        conflicts = memory_contradictions()
        if conflicts:
            db.goals_add(f"Reconcile conflicting facts in '{conflicts[0]['prefix']}'")
            actions.append("conflict_detected")

    if ctr % meta_interval == 0:
        memory_consolidate()
        actions.append("meta_reflection")

    return {"ok": True, "tick": ctr, "actions": actions}

# -----------------------------------------------------
#  Feedback / Meta / Emergence
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
    return {
        "emergent_principles": births,
        "policy_stability": variances,
        "avg_reply_score": metrics.get("avg_reply_score", 0),
        "total_interactions": metrics.get("total_replies", 0),
        "current_policy": db.get_policy_vector(),
    }

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
