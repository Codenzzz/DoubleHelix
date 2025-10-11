# api/chat.py
import os, time, json, re, urllib.parse, urllib.request, asyncio
from typing import List, Dict, Any, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.providers import complete_many, reflect_json
try:
    # optional: some deployments don't export rerank
    from backend.providers import rerank  # type: ignore
except Exception:  # pragma: no cover
    rerank = None  # fallback safely

from backend.utils import db

# ✅ Persistent chat memory bridge + saver
from backend.memory import (
    bridge_context as mem_bridge,
    save_chat_turn as mem_save,
    export_state as mem_export,
)

router = APIRouter(prefix="", tags=["chat"])  # we expose /chat at root

# -----------------------------------------------------
#  Tunables (env)
# -----------------------------------------------------
REQUESTS_PER_MIN = int(os.getenv("REQUESTS_PER_MIN", "20"))
TOKENS_PER_MIN   = int(os.getenv("TOKENS_PER_MIN", "12000"))

# clamp N_SAMPLES to 1..6 (defense in depth)
N_SAMPLES_DEFAULT = max(1, min(6, int(os.getenv("N_SAMPLES", "3"))))

# Heavy prompts (P2/P3) sampling + bridge bounds
HEAVY_NSAMPLES            = max(1, min(3, int(os.getenv("HEAVY_NSAMPLES", "1"))))
ALLOW_PERSPECTIVES_HEAVY  = os.getenv("ALLOW_PERSPECTIVES_HEAVY", "false").lower() == "true"

# Context construction limits
CTX_K_FACTS       = max(0, int(os.getenv("CTX_K_FACTS", "7")))
CTX_K_EMERGENT    = max(0, int(os.getenv("CTX_K_EMERGENT", "2")))
CTX_K_SEARCH_TOP  = max(1, int(os.getenv("CTX_K_SEARCH_TOP", "12")))

# Provider timeout (soft)
PROVIDER_TIMEOUT_MS = max(1000, int(os.getenv("PROVIDER_TIMEOUT_MS", "30000")))

# History/candidates storage cap
MAX_CANDIDATES_STORED = max(1, int(os.getenv("MAX_CANDIDATES_STORED", "3")))

# Dry-run for web
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# -----------------------------------------------------
#  Local rate limiting (chat-only, process-local)
# -----------------------------------------------------
_window_start = time.time()
_req_count = 0
_token_count = 0
_rl_lock = asyncio.Lock()

async def _maybe_rate_limit(tokens_estimate: int = 0):
    global _window_start, _req_count, _token_count
    async with _rl_lock:
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
        _token_count += max(0, int(tokens_estimate))

# -----------------------------------------------------
#  Tiny HTTP + DDG helpers (scoped to chat tools)
# -----------------------------------------------------
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

def _http_get_blocking(url: str, timeout: int = 12) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

async def _http_get(url: str, timeout: int = 12) -> bytes:
    # offload blocking urllib to a thread so we don't block the event loop
    return await asyncio.to_thread(_http_get_blocking, url, timeout)

async def _ddg_html_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    try:
        q = urllib.parse.quote(query)
        url = f"https://duckduckgo.com/html/?q={q}"
        html = (await _http_get(url, timeout=12)).decode("utf-8", errors="ignore")
        items: List[Dict[str, str]] = []
        for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S):
            href = m.group(1)
            title_raw = m.group(2)
            title = re.sub("<.*?>", "", title_raw)
            title = urllib.parse.unquote(re.sub(r"\s+", " ", title)).strip()
            block_start = m.start()
            snippet_match = re.search(r'class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</', html[block_start:block_start+1500], flags=re.I | re.S)
            snippet = ""
            if snippet_match:
                snippet = re.sub("<.*?>", "", snippet_match.group(1))
                snippet = re.sub(r"\s+", " ", snippet).strip()
            items.append({"title": title[:200], "url": href, "snippet": snippet[:280]})
            if len(items) >= limit:
                break
        return items
    except Exception:
        return []

# -----------------------------------------------------
#  History (read-only append used by chat)
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

def _last_chosen_text() -> str:
    """Safely fetch the last assistant reply text (if present)."""
    for it in reversed(_history_get()):
        ch = it.get("chosen")
        if isinstance(ch, str) and ch.strip():
            return ch
    return ""

# -----------------------------------------------------
#  Style + context (chat-scoped)
# -----------------------------------------------------
def _context_from_memory(prompt: str, k: int = 5) -> Tuple[str, Dict[str,int]]:
    # fetch more candidates then filter by lexical overlap with prompt
    cand = db.search_facts(prompt, limit=CTX_K_SEARCH_TOP) or db.top_facts(limit=CTX_K_SEARCH_TOP)
    pw = set(re.findall(r"[a-zA-Z]{3,}", (prompt or "").lower()))
    def ok(f):
        vw = set(re.findall(r"[a-zA-Z]{3,}", str(f.get('value',"")).lower()))
        return len(pw & vw) >= 1
    relevant = [f for f in cand if ok(f)][:k]
    lines = [f"- {r['key']}: {r['value']}" for r in relevant]
    text = ("Context facts:\n" + "\n".join(lines) + "\n\n") if lines else ""
    stats = {"facts_considered": len(relevant)}
    return text, stats

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
    prefs.append('If web info helps, respond with {"tool":"web_search","tool_input":"QUERY"}.')
    # DRY_RUN awareness
    prefs.append('Only call tools if necessary; if web search is disabled (DRY_RUN), explain using current context instead.')
    return "Style preferences: " + "; ".join(prefs) + "."

def get_style_prompt(vec: Dict[str, float]) -> str:
    evolved = db.kv_get("prompts.style_hint")
    if evolved and evolved.get("text"):
        return evolved["text"]
    return _style_hint(vec)

# -----------------------------------------------------
#  Policy scoring (chat-scoped)
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

# -----------------------------------------------------
#  Tool calls (chat-scoped)
# -----------------------------------------------------
def _maybe_tool_call(text: str):
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj and "tool_input" in obj:
            return True, obj
    except Exception:
        pass
    return False, {}

async def _run_tool(obj: Dict[str, Any]) -> str:
    tool, arg = obj.get("tool"), str(obj.get("tool_input", "")).strip()

    # Calculator
    if tool == "calculator":
        if re.search(r"[A-Za-z]", arg) or re.fullmatch(r"\s*\d{4}\s*-\s*\d{2}\s*-\s*\d{2}\s*\s*", arg) or (":" in arg and re.search(r"\d", arg)):
            return "calc_error: natural language or date input ignored"
        try:
            if re.fullmatch(r"[0-9\s+\-*/().]+", arg):
                return str(eval(arg, {"__builtins__": {}}))
        except Exception as e:
            return f"calc_error: {e}"
        return "calc_error: invalid input"

    # Memory search
    if tool == "memory_search":
        return json.dumps({"memory_hits": db.search_facts(arg, limit=5)})

    # Web search
    if tool == "web_search":
        if DRY_RUN:
            return "web_search disabled (DRY_RUN)"

        try:
            query = arg.strip()
            query = re.sub(r'^(?:the\s+web\s+for|the\s+internet\s+for|search\s+the\s+web\s+for)\s+', '', query, flags=re.I)
            query = re.sub(r'\s+', ' ', query).strip()
            if not query:
                return "web_search_error: empty query"

            # DDG Instant Answer (no auth)
            q = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
            try:
                data_raw = await _http_get(url, timeout=12)
                data = json.loads(data_raw.decode("utf-8", errors="ignore"))
            except Exception:
                data = {}

            results: List[Dict[str, str]] = []

            abstract = (data.get("AbstractText") or "").strip() if isinstance(data, dict) else ""
            if abstract:
                results.append({
                    "title": (data.get("Heading") or "Summary") if isinstance(data, dict) else "Summary",
                    "url": (data.get("AbstractURL") or "") if isinstance(data, dict) else "",
                    "snippet": abstract
                })

            related = data.get("RelatedTopics") if isinstance(data, dict) else []
            related = related or []
            for item in related:
                if isinstance(item, dict) and "Topics" in item and isinstance(item["Topics"], list):
                    for t in item["Topics"]:
                        if isinstance(t, dict) and ("FirstURL" in t or "Text" in t):
                            results.append({
                                "title": (t.get("Text") or "").split(" - ")[0][:120],
                                "url": t.get("FirstURL") or "",
                                "snippet": (t.get("Text") or "")[:280]
                            })
                elif isinstance(item, dict) and ("FirstURL" in item or "Text" in item):
                    results.append({
                        "title": (item.get("Text") or "").split(" - ")[0][:120],
                        "url": item.get("FirstURL") or "",
                        "snippet": (item.get("Text") or "")[:280]
                    })

            if not results:
                results = await _ddg_html_results(query, limit=5)

            if not results:
                results = [{"title": "No results", "url": "", "snippet": ""}]

            payload = {"results": results[:5]}

            # Optional rerank with providers.rerank if available
            if rerank and results:
                try:
                    ranked = rerank(query, [r["snippet"] or r["title"] for r in results])[:5]
                    if ranked:
                        payload = {"results": ranked}
                except Exception:
                    pass

            return json.dumps(payload)

        except Exception as e:
            return f"web_search_error: {e}"

    return f"unknown_tool: {tool}"

# -----------------------------------------------------
#  Memory-awareness query detector (chat-scoped)
# -----------------------------------------------------
def _is_memory_awareness_query(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return False

    # NEW: explicit diagnostics also count as memory-awareness
    if t.startswith("memory check") or t.startswith("memory status"):
        return True
    if t in {"memory", "memory?", "check memory", "memory diag", "memory diagnostic"}:
        return True

    if len(t.split()) > 16:
        return False

    t = t.replace("persistant", "persistent")
    if (
        "persistent memory" in t
        or "long term memory" in t
        or "long-term memory" in t
        or "memory status" in t
        or "memory enabled" in t
        or "memory on" in t
        or "do you have memory" in t
        or "do you have persistent" in t
        or "retain memory" in t
        or "save memory" in t
        or "store memory" in t
        or "chat memory" in t
    ):
        return True
    patterns = [
        r"^can\s+you\s+remember",
        r"^do\s+you\s+remember",
        r"^what('?s| is)\s+your\s+memory",
        r"^tell\s+me\s+(your\s+)?memory\s+status",
        r"^how\s+many\s+chats\s+do\s+you\s+remember",
        r"^do\s+you\s+retain\s+memory",
        r"^are\s+you\s+able\s+to\s+remember",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False

# -----------------------------------------------------
#  Perspectives (chat-scoped)
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
#  Lightweight intent router + referent resolver + topic tracker
# -----------------------------------------------------
def classify_intent(text: str) -> str:
    t = (text or "").lower().strip()
    if re.search(r"^\s*(is|was)\s+.+(helpful|useful|working|good)\??$", t): return "opinion"
    if re.search(r"\b(status|enabled|how many|persistent memory|do you remember|memory)\b", t): return "status"
    if re.search(r"^(how|what|give me|write|build|code|plan)\b", t): return "instruction"
    return "chat"

def resolve_referent(prompt: str, last_reply: str) -> str:
    p = (prompt or "").strip()
    low = p.lower()
    if re.fullmatch(r"(is|was)\s+it\s+(helpful|working|good|useful)\??", low):
        m = re.search(r"(persistent memory|continuity|bridge|goal|policy vector|emergent principle)", last_reply or "", re.I)
        subject = m.group(1) if m else "the previous topic"
        pred = low.split()[-1]  # e.g., 'helpful?'
        return f"Is {subject} {pred}?"
    return p

# -----------------------------------------------------
#  Prompt taxonomy (predictive latency classes)
# -----------------------------------------------------
def classify_prompt_class(cmd: str) -> str:
    """
    P0-fast     : very short greetings/status
    P1-normal   : general Q&A
    P2-reflect  : memory/anchor/continuity questions (may include bridge)
    P3-heavy    : explicit diagnostics or very introspective phrasing
    """
    low = (cmd or "").lower().strip()
    if len(low.split()) <= 2:
        return "P0"
    if _is_memory_awareness_query(low):
        return "P2"
    if re.search(r"\b(anchors?|emergent|continuity|bridge|what.*guid(?:e|ing).*(reasoning))\b", low):
        return "P3"
    if re.match(r"^(search|web)\s+", low):
        return "P1"
    return "P1"

# -----------------------------------------------------
#  Chat endpoint
# -----------------------------------------------------
class ChatPayload(BaseModel):
    prompt: str
    model: Optional[str] = None
    use_perspectives: bool = False

async def _call_provider(
    messages: List[Dict[str,str]],
    model: Optional[str],
    style_hint: str,
    n: int = 1
) -> Tuple[str, Dict[str,Any]]:
    """
    Standardize provider call: awaitable, returns (reply, meta).
    Ensures the required `n` is always passed to `complete_many`.
    """
    try:
        reply, meta = await complete_many(messages, n=n, model=model or "gpt-4o-mini", style_hint=style_hint)
        if not isinstance(meta, dict):
            meta = {}
        return str(reply or ""), meta
    except TypeError:
        # In case your provider exposes a sync signature in some envs
        out = complete_many(messages, n=n, model=model or "gpt-4o-mini", style_hint=style_hint)
        # Accept common shapes from different provider adapters
        if isinstance(out, tuple) and len(out) == 2:
            return str(out[0] or ""), dict(out[1] or {})
        if isinstance(out, list) and out:
            # assume list of candidates with {"content": "..."}
            return str(out[0].get("content","")), {}
        return "[model_error] unsupported provider return", {}

@router.post("/chat")
async def chat(p: ChatPayload):
    t0 = time.perf_counter()
    try:
        await _maybe_rate_limit(max(50, len(p.prompt)//3))

        # === JSON tool pass-through ===
        try:
            _as_json = json.loads(p.prompt)
            if isinstance(_as_json, dict) and "tool" in _as_json and "tool_input" in _as_json:
                tool_result = await _run_tool(_as_json)
                total_ms = int((time.perf_counter()-t0)*1000)
                return {
                    "reply": f"(Tool {_as_json['tool']} -> {tool_result})",
                    "meta": {"handled": "direct_tool_call", "route_class": "tool_passthrough", "t_total_ms": total_ms}
                }
        except Exception:
            pass
        # === end pass-through ===

        cmd = p.prompt.strip()
        route_class = "base"
        timings: Dict[str,int] = {}
        meta_extra: Dict[str,Any] = {}
        db_reads = 0

        # === Memory diagnostic command (explicit) ===
        low = cmd.lower()
        if low.startswith("memory check") or low.startswith("memory status") or low in {"memory", "check memory", "memory diag", "memory diagnostic"}:
            route_class = "diagnostic"
            # 1) Write a ping fact (best-effort)
            try:
                mem_save('memory_check', 'ping test memory', meta={"source": "diagnostic"})
            except Exception:
                pass

            # 2) Read the last 2 facts (best-effort)
            try:
                recent_all = db.all_facts() or []; db_reads += 1
                recent_all_sorted = sorted(recent_all, key=lambda x: x.get("ts",""))
                recent_two = recent_all_sorted[-2:] if recent_all_sorted else []
                last_two = [{"key": f.get("key"), "value": f.get("value"), "confidence": f.get("confidence", 0.0)} for f in recent_two]
            except Exception:
                last_two = []

            # 3) Summaries
            try:
                total_facts = len(db.all_facts() or []); db_reads += 1
            except Exception:
                total_facts = None

            last_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            total_ms = int((time.perf_counter()-t0)*1000)

            return {
                "reply": (
                    "✅ Memory diagnostic\n"
                    f"- Wrote: 'ping test memory'\n"
                    f"- Last 2 facts: {last_two}\n"
                    f"- Total facts: {total_facts}\n"
                    f"- last_ts: {last_ts}"
                ),
                "meta": {
                    "handled": "memory_check",
                    "route_class": route_class,
                    "total_facts": total_facts,
                    "last_two": last_two,
                    "last_ts": last_ts,
                    "t_total_ms": total_ms,
                    "db_reads": db_reads
                }
            }
        # === end memory diagnostic ===

        # "search ..." or "web ..." -> force web_search tool
        m1 = re.match(r'^(?:search|web)\s+(.+)$', cmd, flags=re.I)
        if m1:
            route_class = "web"
            q = m1.group(1).strip()
            tool_result = await _run_tool({"tool": "web_search", "tool_input": q})
            total_ms = int((time.perf_counter()-t0)*1000)
            return {
                "reply": f"(Tool web_search -> {tool_result})",
                "meta": {"handled": "command_web_search", "route_class": route_class, "query": q, "t_total_ms": total_ms}
            }

        # Resolve vague referents using last assistant reply (safe)
        last_reply = _last_chosen_text()
        cmd = resolve_referent(cmd, last_reply)

        # /memory awareness (uses memory.export_state)
        if _is_memory_awareness_query(cmd):
            route_class = "memory_awareness"
            st = mem_export();  # may be heavy
            enabled = bool(st.get("enabled"))
            turns = int((st.get("stats") or {}).get("turns", 0))
            recent_n = int((st.get("recent") or {}).get("n", 0))
            recent_items = (st.get("recent") or {}).get("items", [])[-3:]
            lines = []
            for it in recent_items:
                pr = str(it.get("prompt","")).replace("\n"," ")[:70]
                rp = str(it.get("reply","")).replace("\n"," ")[:80]
                ts = it.get("ts","")
                lines.append(f"- {ts}  Q:{pr} | A:{rp}")
            summary = "\n".join(lines) if lines else "(no recent snapshots yet)"
            total_ms = int((time.perf_counter()-t0)*1000)
            return {
                "reply": (
                    "Yes — I persist chat turns now.\n"
                    f"Enabled: {enabled} | Stored turns: {turns} | Recent window: {recent_n}\n"
                    f"Recent chat snapshots:\n{summary}"
                ),
                "meta": {"handled": "memory_awareness", "route_class": route_class, "memory_enabled": enabled, "turns": turns, "recent_n": recent_n, "t_total_ms": total_ms}
            }

        # ----- Normal model path -----
        t_ctx0 = time.perf_counter()

        vec = db.get_policy_vector(); db_reads += 1
        intent = classify_intent(cmd)

        # intent-conditioned style + ambiguity hint for tiny follow-ups
        style = get_style_prompt(vec)
        style += " " + {
            "status": "Answer with a factual status about Helix internals.",
            "opinion": "Give a concise, first-person assessment grounded in Helix state.",
            "instruction": "Provide a step-by-step actionable answer.",
            "chat": "Be natural and concise.",
        }[intent]
        low_context = (len(cmd.split()) < 4)
        if low_context:
            style += " If the user question is ambiguous, say what you think 'it' refers to before answering."

        memory = db.top_facts(5); db_reads += 1

        # Goal-aware + Continuity Bridge context
        goal = db.goal_active(); db_reads += 1
        goal_text = goal.get("text", "") if isinstance(goal, dict) else (goal or "")
        goal_prefix = f"Active goal: {goal_text}\n\n" if goal_text else ""

        # Reality Bridge (local) — bounded by env limits
        def _bridge_context_local(k_facts: int = CTX_K_FACTS, k_emergent: int = CTX_K_EMERGENT) -> Tuple[str, Dict[str,int], List[str]]:
            facts = db.top_facts(limit=20); 
            # count once here as a read; above db_reads is approximate visibility only
            picked_keys: List[str] = []
            emergent_keys: List[str] = []
            if not isinstance(facts, list):
                facts = []
            facts_sorted = sorted(facts, key=lambda x: float(x.get("confidence", 0.5)), reverse=True)
            picked = facts_sorted[:k_facts]
            emergent = [f for f in facts_sorted if str(f.get("key","")).startswith("emergent:")][:k_emergent]
            lines = []
            components = []
            if goal_text:
                lines.append(f"Active goal: {goal_text}")
                components.append("goal")
            if picked:
                lines.append("Stable anchors:")
                for f in picked:
                    lines.append(f"  - {f['key']}: {f['value']}")
                    picked_keys.append(str(f.get("key","")))
                components.append("facts")
            if emergent:
                lines.append("Recent emergent principles:")
                for e in emergent:
                    lines.append(f"  - {e['key']}: {e['value']}")
                    emergent_keys.append(str(e.get("key","")))
                components.append("emergent")
            text = ("\n".join(lines) + "\n\n") if lines else ""
            stats = {"bridge_facts": len(picked), "bridge_emergent": len(emergent)}
            return text, stats, components

        def _should_bridge_local() -> bool:
            thresholds = db.kv_get("thresholds") or {}; 
            bridge_s = float(thresholds.get("bridge_surprise", 0.65))
            bridge_v = float(thresholds.get("bridge_var", 0.03))
            last_surprise = float((db.kv_get("last.surprise") or {}).get("value", 0.0))
            # reuse policy var calc lightly
            hist = db.get_policy_history(10); 
            variances: Dict[str, float] = {}
            if len(hist) >= 2:
                for trait in ("creativity","conciseness","planning_focus","skepticism"):
                    vals = [float(h.get(trait,0.0)) for h in hist if isinstance(h,dict)]
                    if len(vals) >= 2:
                        m = sum(vals)/len(vals)
                        variances[trait] = sum((v-m)**2 for v in vals)/len(vals)
            max_var = max(variances.values()) if variances else 0.0
            return last_surprise >= bridge_s or max_var >= bridge_v

        bridge_text = ""
        bridge_stats: Dict[str,int] = {}
        bridge_components: List[str] = []

        if _should_bridge_local():
            route_class = "bridge_local"
            bridge_text, bridge_stats, bridge_components = _bridge_context_local()

        continuity = mem_bridge(cmd)  # may be empty string
        if continuity:
            bridge_components = sorted(set(bridge_components + ["continuity"]))

        # Topic tracker (store after we answer; also read latest topic here)
        thread = db.kv_get("thread.topic") or {}; 
        topic_line = f"Current topic: {thread.get('value')}\n\n" if thread.get('value') else ""

        # Short follow-up bias
        bias = "Follow-up question; answer about the immediately prior assistant reply.\n\n" if low_context else ""

        # Memory facts (lexical overlap filter)
        ctx_mem_text, mem_stats = _context_from_memory(cmd, k=5)

        # Final context
        context = bias + topic_line + goal_prefix + bridge_text + (continuity or "") + ctx_mem_text
        context_bytes = len(context.encode("utf-8"))

        timings["t_context_ms"] = int((time.perf_counter()-t_ctx0)*1000)

        # === Candidate generation ===
        messages = [{"role": "user", "content": context + cmd}]
        candidates: List[Tuple[float,str,Dict[str,float]]] = []
        profiles_used: List[str] = []

        prompt_class = classify_prompt_class(cmd)

        # Adaptive sampling: heavy prompts should not fan out
        if prompt_class in ("P2","P3"):
            nsamples = HEAVY_NSAMPLES
            use_persp = p.use_perspectives and ALLOW_PERSPECTIVES_HEAVY
        else:
            nsamples = N_SAMPLES_DEFAULT
            use_persp = p.use_perspectives

        # provider call(s)
        t_model0 = time.perf_counter()
        provider_timeout = PROVIDER_TIMEOUT_MS / 1000.0

        async def _one_call(_style, _vec):
            return await _call_provider(messages, p.model, _style, n=1)

        if use_persp:
            for profile in ["base", "explorer", "skeptic", "planner"]:
                p_vec = perturb_policy(vec, profile) if profile != "base" else vec
                p_style = get_style_prompt(p_vec) + " " + style
                # soft timeout guard
                reply_text, meta0 = await asyncio.wait_for(_one_call(p_style, p_vec), timeout=provider_timeout)
                s, subs = _score_with_policy(reply_text, memory, p_vec)
                candidates.append((s, reply_text, subs))
                profiles_used.append(profile)
        else:
            for _ in range(nsamples):
                reply_text, meta0 = await asyncio.wait_for(_one_call(style, vec), timeout=provider_timeout)
                s, subs = _score_with_policy(reply_text, memory, vec)
                candidates.append((s, reply_text, subs))
            profiles_used.append("base")

        timings["t_provider_ms"] = int((time.perf_counter()-t_model0)*1000)

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text, best_subs = candidates[0]
        reply_bytes = len(best_text.encode("utf-8"))

        # Topic tracking: pull a lightweight topic from best_text and store it for next turn
        topic_match = re.search(r"(persistent memory|continuity|bridge|policy(?: vector)?|goals?|illusion|clean eyes)", best_text, re.I)
        if topic_match:
            db.kv_upsert("thread.topic", {"value": topic_match.group(1), "ts": time.strftime("%Y-%m-%d %H:%M:%S")})

        # Policy nudge & persist
        vec_after = _policy_nudge(vec, best_subs, 0.06)
        db.set_policy_vector(vec_after)

        # Metrics + surprise tracking
        metrics = db.kv_get("metrics") or {}
        prev_total = int(metrics.get("total_replies", 0))
        new_total = prev_total + 1
        prev_avg = float(metrics.get("avg_reply_score", 0.0))
        metrics["total_replies"] = new_total
        metrics["avg_reply_score"] = round((prev_avg * prev_total + best_score) / new_total, 3)
        db.kv_upsert("metrics", metrics)

        surprise = float(best_subs.get("creativity", 0.0))
        db.kv_upsert("last.surprise", {"value": surprise})

        # History (cap stored candidates for size/latency control)
        stored_candidates = [t for _, t, _ in candidates[:MAX_CANDIDATES_STORED]]
        t_hist0 = time.perf_counter()
        _history_append({
            "kind": "chat",
            "prompt": p.prompt,     # keep original user text
            "normalized": cmd,      # what we actually used after referent resolution
            "active_goal": goal_text,
            "used_perspectives": use_persp,
            "profiles": profiles_used,
            "candidates": stored_candidates,
            "chosen": best_text,
            "score": best_score,
            "bridge_used": bool(bridge_text or continuity),
            "bridge_components": bridge_components,
            "surprise": surprise
        })
        timings["t_history_ms"] = int((time.perf_counter()-t_hist0)*1000)

        # ✅ Persist this chat turn to memory (best-effort)
        try:
            mem_save(
                p.prompt,            # store original prompt
                best_text,
                meta={
                    "policy_before": vec,
                    "policy_after": vec_after,
                    "score": best_score,
                    "surprise": surprise,
                    "profiles": profiles_used,
                    "normalized": cmd,   # store normalized too (useful for analysis)
                    "intent": classify_intent(cmd),
                    "prompt_class": prompt_class
                }
            )
        except Exception:
            pass

        total_ms = int((time.perf_counter()-t0)*1000)

        return {
            "reply": best_text,
            "meta": {
                # Core score & policy
                "score": best_score,
                "surprise": round(surprise, 3),
                "policy": vec_after,
                # Production route labeling
                "handled": "model",
                "route_class": route_class,
                "intent": intent,
                "prompt_class": prompt_class,
                # Context + bridge
                "bridge_used": bool(bridge_text or continuity),
                "bridge_components": bridge_components,
                "facts_considered": mem_stats.get("facts_considered", 0),
                "bridge_facts": bridge_stats.get("bridge_facts", 0),
                "bridge_emergent": bridge_stats.get("bridge_emergent", 0),
                "context_bytes": context_bytes,
                "reply_bytes": reply_bytes,
                # Sampling
                "candidates": len(candidates),
                "candidates_stored": len(stored_candidates),
                "used_perspectives": bool(use_persp),
                "profiles": profiles_used,
                "samples_requested": nsamples,
                "sampling_mode": "perspectives" if use_persp else "multi_sample",
                # Timings
                "t_total_ms": total_ms,
                **timings,
                # Health-ish hints
                "db_reads": db_reads,
            }
        }

    except asyncio.TimeoutError:
        total_ms = int((time.perf_counter()-t0)*1000)
        return {"reply": "[timeout] provider exceeded time budget", "meta": {"error": True, "route_class":"model", "t_total_ms": total_ms}}
    except HTTPException:
        raise
    except Exception as e:
        total_ms = int((time.perf_counter()-t0)*1000)
        return {"reply": f"[server_error] {type(e).__name__}: {e}", "meta": {"error": True, "t_total_ms": total_ms}}

# -----------------------------------------------------
#  Minimal history read API (for frontend)
# -----------------------------------------------------
@router.get("/history")
def api_history(limit: int = 80):
    try:
        items = _history_get()
        return {"history": items[-max(1, min(500, limit)):] }
    except Exception as e:
        raise HTTPException(500, f"history_error: {e}")
