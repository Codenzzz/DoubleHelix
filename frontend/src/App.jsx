// src/App.jsx
import React, { useEffect, useMemo, useState } from "react";

/** ------------ API base resolution ------------ */
const resolvedApiBase =
  (typeof import.meta !== "undefined" &&
    import.meta.env &&
    import.meta.env.VITE_API_BASE) ||
  (typeof window !== "undefined" &&
  window.location.hostname.includes("localhost")
    ? "http://localhost:8000"
    : "https://doublehelix.onrender.com");

// Build absolute URL safely (no double/missing slashes)
function apiURL(path) {
  return new URL(String(path).replace(/^\/*/, "/"), resolvedApiBase).toString();
}

async function getJSON(path, opts) {
  const res = await fetch(apiURL(path), opts);
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    const err = new Error(`GET ${path} → ${res.status} ${res.statusText} ${txt}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

async function postJSON(path, body, opts = {}) {
  const res = await fetch(apiURL(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    ...opts,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`POST ${path} → ${res.status} ${res.statusText} ${txt}`);
  }
  return res.json();
}

async function del(path) {
  const res = await fetch(apiURL(path), { method: "DELETE" });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`DELETE ${path} → ${res.status} ${res.statusText} ${txt}`);
  }
  return res.json();
}

/** ------------ App ------------ */
export default function App() {
  const API_BASE = useMemo(() => resolvedApiBase, []);
  const [health, setHealth] = useState("checking...");
  const [error, setError] = useState(null);

  // 🔹 New: Memory & Clock state
  const [memoryStatus, setMemoryStatus] = useState(null);
  const [clock, setClock] = useState(null);

  // Policy
  const [policyVector, setPolicyVector] = useState({});
  const [policyHistory, setPolicyHistory] = useState([]);
  const [metrics, setMetrics] = useState(null);

  // Goals
  const [goals, setGoals] = useState([]);
  const [newGoal, setNewGoal] = useState("");

  // Emergence
  const [emergence, setEmergence] = useState(null);

  // Clean Eyes
  const [cleanPreview, setCleanPreview] = useState(null);
  const [cleanAlpha, setCleanAlpha] = useState("");

  // History buffer
  const [history, setHistory] = useState([]);

  // Chat state
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const [messages, setMessages] = useState([]);
  const [usePerspectives, setUsePerspectives] = useState(false);
  const [model, setModel] = useState(null);

  /** -------- Memory + History fetch -------- */
  async function fetchMemoryStatus() {
    try {
      const ms = await getJSON("/memory/status");
      setMemoryStatus(ms);
      if (ms?.clock) setClock(ms.clock);
    } catch (e) {
      console.warn("Memory status fetch failed:", e);
    }
  }

  async function pingClock() {
    try {
      const res = await postJSON("/memory/clock", { event: "manual_ping" });
      await fetchMemoryStatus();
      console.log("Clock ping:", res);
    } catch (e) {
      setError(String(e));
    }
  }

  async function fetchHistory() {
    try {
      const h = await getJSON("/history?limit=80");
      if (Array.isArray(h?.history)) return h.history;
      if (Array.isArray(h?.items)) return h.items;
    } catch (e) {
      if (e?.status && e.status !== 404) throw e;
    }
    try {
      const h = await getJSON("/history");
      if (Array.isArray(h?.history)) return h.history;
      if (Array.isArray(h?.items)) return h.items;
    } catch (e) {
      if (e?.status && e.status !== 404) throw e;
    }
    try {
      const ms = await getJSON("/memory/status");
      const latest = Array.isArray(ms?.latest) ? ms.latest : [];
      return latest.map((f) => ({
        ts: "",
        kind: "memory_latest",
        key: f.key,
        value: f.value,
        confidence: f.confidence,
      }));
    } catch {
      return [];
    }
  }

  async function refresh() {
    setError(null);

    try {
      const d = await getJSON("/health");
      setHealth(d.status ?? "ok");
    } catch {
      setHealth("error");
    }

    try {
      const pv = await getJSON("/policy");
      setPolicyVector(pv || {});
    } catch {}

    try {
      const ph = await getJSON("/policy/history?limit=40");
      setPolicyHistory(ph?.history || []);
    } catch {}

    try {
      const g = await getJSON("/goals");
      setGoals(g?.goals || []);
    } catch {}

    try {
      const em = await getJSON("/emergence");
      setEmergence(em);
      setMetrics({
        avg_reply_score: em?.avg_reply_score,
        total_interactions: em?.total_interactions,
        policy_stability: em?.policy_stability,
      });
    } catch {}

    try {
      const items = await fetchHistory();
      setHistory(items || []);
    } catch {
      setHistory([]);
    }

    await fetchMemoryStatus();
  }

  useEffect(() => {
    refresh();
    const timer = setInterval(() => fetchMemoryStatus(), 30000); // auto-refresh every 30s
    return () => clearInterval(timer);
  }, []);

  async function addGoal() {
    if (!newGoal.trim()) return;
    try {
      await postJSON("/goals", { text: newGoal });
      setNewGoal("");
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function activateGoal(id) {
    try {
      await fetch(apiURL(`/goals/${id}/activate`), { method: "POST" });
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function deleteGoal(id) {
    try {
      await del(`/goals/${id}`);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function dedupeGoals() {
    try {
      await postJSON("/goals/dedupe");
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function compactMemory() {
    try {
      await postJSON("/memory/compact", {});
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function tick() {
    try {
      await fetch(apiURL("/tick"), { method: "POST" });
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function previewCleanEyes() {
    try {
      const ce = await getJSON(
        cleanAlpha ? `/clean/preview?alpha=${encodeURIComponent(cleanAlpha)}` : "/clean/preview"
      );
      setCleanPreview(ce || null);
    } catch (e) {
      setError(String(e));
    }
  }

  async function applyCleanEyes() {
    try {
      const url = cleanAlpha
        ? `/clean/apply?alpha=${encodeURIComponent(cleanAlpha)}`
        : "/clean/apply";
      await postJSON(url, undefined);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function sendChat(e) {
    e?.preventDefault?.();
    const prompt = chatInput.trim();
    if (!prompt || chatSending) return;
    setChatSending(true);
    setMessages((m) => [...m, { role: "user", text: prompt }]);
    setChatInput("");
    try {
      const resp = await postJSON("/chat", {
        prompt,
        model,
        use_perspectives: usePerspectives,
      });
      setMessages((m) => [
        ...m,
        { role: "assistant", text: resp.reply ?? "(no reply)", meta: resp.meta },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", text: `⚠️ ${String(err)}` }]);
    } finally {
      setChatSending(false);
    }
  }

  function onChatKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) sendChat(e);
  }

  const activeGoalObj = goals.find((g) => g.active) || null;

  return (
    <div className="container">
      <div className="title">🧬 DoubleHelix v0.9.1</div>
      <div className="subtitle">
        Emergence • Continuity • Reality Bridge • Temporal Awareness
      </div>

      {/* Quick Controls */}
      <div className="row" style={{ marginBottom: 8, gap: 8, flexWrap: "wrap" }}>
        <span className="pill">API: {API_BASE}</span>
        <span className="pill">health: {health}</span>
        {memoryStatus?.enabled ? (
          <span className="pill" style={{ background: "#2a6" }}>memory: on</span>
        ) : (
          <span className="pill" style={{ background: "#633" }}>memory: off</span>
        )}
        {error && <span className="pill" style={{ background: "#522" }}>err</span>}
        <button className="pill" onClick={refresh}>Refresh</button>
        <button className="pill" onClick={tick}>Planner Tick</button>
        <button className="pill" onClick={pingClock}>Ping Clock</button>
      </div>

      {/* Memory Clock Panel */}
      {clock && (
        <div className="card" style={{ borderColor: "#39f" }}>
          <h3>Memory Clock</h3>
          <div className="row" style={{ flexWrap: "wrap", gap: 8 }}>
            <span className="pill">last event: {clock.last_event || "—"}</span>
            <span className="pill">Δt: {clock.delta_sec ?? "—"}s</span>
            <span className="pill">history: {clock.history_len}</span>
          </div>
          <div className="muted" style={{ marginTop: 6 }}>
            Auto-refreshes every 30s to show temporal continuity drift.
          </div>
        </div>
      )}

      {/* Goals */}
      <div className="card">
        <h3>Goals</h3>
        <div className="row" style={{ gap: 8 }}>
          <input
            placeholder="New goal…"
            value={newGoal}
            onChange={(e) => setNewGoal(e.target.value)}
            style={{ flex: 1 }}
          />
          <button onClick={addGoal}>Add</button>
          <button className="ghost" title="Remove duplicate goals" onClick={dedupeGoals}>
            Dedupe
          </button>
        </div>
        <div style={{ marginTop: 10 }}>
          {goals.map((g) => (
            <div key={g.id} className="row" style={{ marginBottom: 6, gap: 8 }}>
              <span className="pill">#{g.id}</span>
              <span style={{ flex: 1 }}>{g.text}</span>
              <span className="pill">{g.active ? "active" : "inactive"}</span>
              {!g.active && (
                <button className="ghost" onClick={() => activateGoal(g.id)}>
                  Activate
                </button>
              )}
              <button className="ghost" onClick={() => deleteGoal(g.id)} title="Delete goal">
                Delete
              </button>
            </div>
          ))}
          {!goals.length && <div className="muted">No goals yet. Add one above.</div>}
        </div>
      </div>

      {/* Emergence */}
      <div className="card">
        <h3>Emergence</h3>
        <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
          <span className="pill">
            births: {emergence?.emergent_principles ?? 0}
          </span>
          <span className="pill">
            surprise: {typeof emergence?.surprise === "number" ? emergence.surprise.toFixed(3) : "-"}
          </span>
          <button className="pill" onClick={compactMemory} title="Deduplicate and prune facts">
            Compact Memory
          </button>
        </div>
        {!!emergence?.illusion && (
          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: "pointer" }}>illusion (dream/recall)</summary>
            <pre style={{ margin: 0 }}>
              {JSON.stringify(emergence.illusion, null, 2)}
            </pre>
          </details>
        )}
        <details style={{ marginTop: 8 }}>
          <summary style={{ cursor: "pointer" }}>raw emergence</summary>
          <pre style={{ margin: 0 }}>
            {emergence ? JSON.stringify(emergence, null, 2) : "(no data yet)"}
          </pre>
        </details>
      </div>

      {/* Clean Eyes */}
      <div className="card">
        <h3>Clean Eyes (policy-only)</h3>
        <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
          <input
            className="pill"
            style={{ width: 120 }}
            placeholder="alpha (0..1)"
            value={cleanAlpha}
            onChange={(e) => setCleanAlpha(e.target.value)}
          />
          <button className="pill" onClick={previewCleanEyes}>Preview</button>
          <button className="pill" onClick={applyCleanEyes}>Apply</button>
        </div>
        <div style={{ marginTop: 8 }}>
          {cleanPreview ? (
            <pre style={{ maxHeight: 220, overflow: "auto" }}>
              {JSON.stringify(cleanPreview, null, 2)}
            </pre>
          ) : (
            <div className="muted">No preview yet.</div>
          )}
        </div>
        <div className="muted">Non-destructive: resets behavior only, not facts/goals.</div>
      </div>

      {/* History */}
      <div className="card">
        <h3>History (recent)</h3>
        <div style={{ maxHeight: 240, overflow: "auto", border: "1px solid #333", borderRadius: 8, padding: 8 }}>
          {!history?.length && <div className="muted">(empty)</div>}
          {history?.slice().reverse().map((h, i) => (
            <details key={i} style={{ marginBottom: 8 }}>
              <summary style={{ cursor: "pointer" }}>
                <code>{h.ts || "—"}</code> — <em>{h.kind || "entry"}</em>
              </summary>
              <pre style={{ margin: 0 }}>{JSON.stringify(h, null, 2)}</pre>
            </details>
          ))}
        </div>
      </div>

      {/* Chat */}
      <div className="card">
        <h3>Chat</h3>
        <div className="row" style={{ gap: 8, marginBottom: 8, flexWrap: "wrap" }}>
          <label className="pill" style={{ cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={usePerspectives}
              onChange={(e) => setUsePerspectives(e.target.checked)}
              style={{ marginRight: 6 }}
            />
            use perspectives
          </label>
          <select
            value={model ?? ""}
            onChange={(e) => setModel(e.target.value || null)}
            className="pill"
            title="Model (optional; leave blank to use backend default)"
          >
            <option value="">default (backend)</option>
            <option value="gpt-4o-mini">gpt-4o-mini</option>
            <option value="gpt-4o">gpt-4o</option>
             <option value="gpt-5">gpt-5</option> 
          </select>
        </div>

        <div
          style={{
            border: "1px solid #333",
            borderRadius: 8,
            padding: 8,
            height: 260,
            overflow: "auto",
            background: "#0b0d10",
          }}
        >
          {messages.length === 0 && (
            <div className="muted">(No messages yet. Say hello 👋)</div>
          )}
          {messages.map((m, i) => (
            <div key={i} style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 12, opacity: 0.7 }}>{m.role}</div>
              <div
                style={{
                  whiteSpace: "pre-wrap",
                  background: m.role === "user" ? "#1b1f24" : "#0f1720",
                  border: "1px solid #2a2f36",
                  padding: 8,
                  borderRadius: 8,
                }}
              >
                {m.text}
              </div>
              {m.meta && (
                <details style={{ marginTop: 6 }}>
                  <summary style={{ cursor: "pointer" }}>meta</summary>
                  <pre style={{ margin: 0 }}>
                    {JSON.stringify(m.meta, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          ))}
        </div>

        <form onSubmit={sendChat} className="row" style={{ marginTop: 8, gap: 8 }}>
          <input
            placeholder="Type a message… (Enter to send)"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={onChatKeyDown}
            disabled={chatSending}
            style={{ flex: 1 }}
          />
          <button disabled={chatSending} className="pill" onClick={sendChat}>
            {chatSending ? "Sending…" : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
}
