# api/chat.py
import os, time, json, re, urllib.parse, urllib.request
from typing import List, Dict, Any, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from providers import complete_many, reflect_json
try:
    # optional: some deployments don't export rerank
    from providers import rerank  # type: ignore
except Exception:  # pragma: no cover
    rerank = None  # fallback safely

from utils import db

# ✅ Persistent chat memory bridge + saver
from memory import (
    bridge_context as mem_bridge,
    save_chat_turn as mem_save,
    export_state as mem_export,
)

router = APIRouter(prefix="", tags=["chat"])  # we expose /chat at root

# -----------------------------------------------------
#  Local rate limiting (chat-only)
# -----------------------------------------------------
REQUESTS_PER_MIN = int(os.getenv("REQUESTS_PER_MIN", "20"))
TOKENS_PER_MIN   = int(os.getenv("TOKENS_PER_MIN", "12000"))
# clamp N_SAMPLES to 1..6 (defense in depth)
N_SAMPLES        = max(1, min(6, int(os.getenv("N_SAMPLES", "3"))))
DRY_RUN          = os.getenv("DRY_RUN", "true").lower() == "true"

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
#  Tiny HTTP + DDG helpers (scoped to chat tools)
# -----------------------------------------------------
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

def _http_get(url: str, timeout: int = 12) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def _ddg_html_results(query: str, limit: int = 5) -> List[Dict[str, str]]:
    try:
        q = urllib.parse.quote(query)
        url = f"https://duckduckgo.com/html/?q={q}"
        html = _http_get(url, timeout=12).decode("utf-8", errors="ignore")
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
#     (includes smarter memory gating)
# -----------------------------------------------------
def _context_from_memory(prompt: str, k: int = 5) -> str:
    # fetch more candidates then filter by lexical overlap with prompt
    cand = db.search_facts(prompt, limit=12) or db.top_facts(limit=12)
    pw = set(re.findall(r"[a-zA-Z]{3,}", (prompt or "").lower()))
    def ok(f):
        vw = set(re.findall(r"[a-zA-Z]{3,}", str(f.get('value',"")).lower()))
        return len(pw & vw) >= 1
    relevant = [f for f in cand if ok(f)][:k]
    lines = [f"- {r['key']}: {r['value']}" for r in relevant]
    return ("Context facts:\n" + "\n".join(lines) + "\n\n") if lines else ""

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

def _run_tool(obj: Dict[str, Any]) -> str:
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
                data = json.loads(_http_get(url, timeout=12).decode("utf-8", errors="ignore"))
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
                results = _ddg_html_results(query, limit=5)

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
    """
    Detect short, direct questions about Helix's memory capabilities/status.
    Covers 'remember' as well as 'have persistent/long-term memory' (incl. common misspelling).
    """
    t = (text or "").lower().strip()
    if not t:
        return False
    if len(t.split()) > 16:
        return False
    t = t.replace("persistant", "persistent")  # normalize common misspelling
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
    """
    Replace vague referents ('it/this/that') with inferred topic from last reply.
    Cheap heuristic: look for known nouns; fall back to 'the previous topic'.
    """
    p = (prompt or "").strip()
    low = p.lower()
    if re.fullmatch(r"(is|was)\s+it\s+(helpful|working|good|useful)\??", low):
        m = re.search(r"(persistent memory|continuity|bridge|goal|policy vector|emergent principle)", last_reply or "", re.I)
        subject = m.group(1) if m else "the previous topic"
        pred = low.split()[-1]  # e.g., 'helpful?'
        return f"Is {subject} {pred}?"
    return p

# -----------------------------------------------------
#  Chat endpoint
# -----------------------------------------------------
class ChatPayload(BaseModel):
    prompt: str
    model: Optional[str] = None
    use_perspectives: bool = False

@router.post("/chat")
def chat(p: ChatPayload):
    try:
        _maybe_rate_limit(max(50, len(p.prompt)//3))

        # === JSON tool pass-through ===
        try:
            _as_json = json.loads(p.prompt)
            if isinstance(_as_json, dict) and "tool" in _as_json and "tool_input" in _as_json:
                tool_result = _run_tool(_as_json)
                return {
                    "reply": f"(Tool {_as_json['tool']} -> {tool_result})",
                    "meta": {"handled": "direct_tool_call"}
                }
        except Exception:
            pass
        # === end pass-through ===

        cmd = p.prompt.strip()

        # "search ..." or "web ..." -> force web_search tool
        m1 = re.match(r'^(?:search|web)\s+(.+)$', cmd, flags=re.I)
        if m1:
            q = m1.group(1).strip()
            tool_result = _run_tool({"tool": "web_search", "tool_input": q})
            return {
                "reply": f"(Tool web_search -> {tool_result})",
                "meta": {"handled": "command_web_search", "query": q}
            }

        # Resolve vague referents using last assistant reply (safe)
        last_reply = _last_chosen_text()
        cmd = resolve_referent(cmd, last_reply)

        # /memory awareness (uses memory.export_state)
        if _is_memory_awareness_query(cmd):
            st = mem_export()
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
            return {
                "reply": (
                    "Yes — I persist chat turns now.\n"
                    f"Enabled: {enabled} | Stored turns: {turns} | Recent window: {recent_n}\n"
                    f"Recent chat snapshots:\n{summary}"
                ),
                "meta": {"handled": "memory_awareness", "memory_enabled": enabled, "turns": turns, "recent_n": recent_n}
            }

        # ----- Normal model path -----
        vec = db.get_policy_vector()
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

        memory = db.top_facts(5)

        # Goal-aware + Continuity Bridge context
        goal = db.goal_active()
        goal_text = goal.get("text", "") if isinstance(goal, dict) else (goal or "")
        goal_prefix = f"Active goal: {goal_text}\n\n" if goal_text else ""

        # Reality Bridge: goal + anchors from facts (local copy)
        def _bridge_context_local(k_goal: int = 1, k_facts: int = 7, k_emergent: int = 2) -> str:
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

        def _should_bridge_local() -> bool:
            thresholds = db.kv_get("thresholds") or {}
            bridge_s = float(thresholds.get("bridge_surprise", 0.65))
            bridge_v = float(thresholds.get("bridge_var", 0.03))
            last_surprise = float((db.kv_get("last.surprise") or {}).get("value", 0.0))
            # reuse policy var calc lightly
            hist = db.get_policy_history(10)
            variances: Dict[str, float] = {}
            if len(hist) >= 2:
                for trait in ("creativity","conciseness","planning_focus","skepticism"):
                    vals = [float(h.get(trait,0.0)) for h in hist if isinstance(h,dict)]
                    if len(vals) >= 2:
                        m = sum(vals)/len(vals)
                        variances[trait] = sum((v-m)**2 for v in vals)/len(vals)
            max_var = max(variances.values()) if variances else 0.0
            return last_surprise >= bridge_s or max_var >= bridge_v

        bridge = _bridge_context_local() if _should_bridge_local() else ""
        continuity = mem_bridge(cmd)

        # Topic tracker (store after we answer; also read latest topic here)
        thread = db.kv_get("thread.topic") or {}
        topic_line = f"Current topic: {thread.get('value')}\n\n" if thread.get('value') else ""

        # Short follow-up bias
        bias = "Follow-up question; answer about the immediately prior assistant reply.\n\n" if low_context else ""

        # Final context
        context = bias + topic_line + goal_prefix + bridge + (continuity or "") + _context_from_memory(cmd, k=5)

        outs = []
        used_profiles = []
        try:
            if p.use_perspectives:
                for profile in ["base", "explorer", "skeptic", "planner"]:
                    p_vec = perturb_policy(vec, profile) if profile != "base" else vec
                    p_style = get_style_prompt(p_vec) + " " + style
                    outs.extend(complete_many(context + cmd, n=1, model=p.model or "gpt-4o-mini", style_hint=p_style))
                    used_profiles.append(profile)
            else:
                outs = complete_many(context + cmd, n=N_SAMPLES, model=p.model or "gpt-4o-mini", style_hint=style)
                used_profiles.append("base")
        except Exception as e:
            # if the provider explodes, fail gracefully with a single candidate
            outs = [{"content": f"[model_error] {type(e).__name__}: {e}"}]
            used_profiles = ["base"]

        processed = []
        for o in outs:
            text = o.get("content", "")
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

        # Topic tracking: pull a lightweight topic from best_text and store it for next turn
        topic_match = re.search(r"(persistent memory|continuity|bridge|policy(?: vector)?|goals?|illusion|clean eyes)", best_text, re.I)
        if topic_match:
            db.kv_upsert("thread.topic", {"value": topic_match.group(1), "ts": time.strftime("%Y-%m-%d %H:%M:%S")})

        # Policy nudge & persist
        vec = _policy_nudge(vec, best_subs, 0.06)
        db.set_policy_vector(vec)

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

        # History
        _history_append({
            "kind": "chat",
            "prompt": p.prompt,     # keep original user text
            "normalized": cmd,      # what we actually used after referent resolution
            "active_goal": goal_text,
            "used_perspectives": p.use_perspectives,
            "profiles": used_profiles,
            "candidates": [t for _, t, _ in scored],
            "chosen": best_text,
            "score": best_score,
            "bridge_used": bool(bridge or continuity),
            "surprise": surprise
        })

        # ✅ Persist this chat turn to memory
        try:
            mem_save(
                p.prompt,            # store original prompt
                best_text,
                meta={
                    "policy": vec,
                    "score": best_score,
                    "surprise": surprise,
                    "profiles": used_profiles,
                    "normalized": cmd,   # store normalized too (useful for analysis)
                    "intent": intent
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
#  Minimal history read API (for frontend)
# -----------------------------------------------------
@router.get("/history")
def api_history(limit: int = 80):
    try:
        items = _history_get()
        return {"history": items[-max(1, min(500, limit)):] }
    except Exception as e:
        raise HTTPException(500, f"history_error: {e}")
