import React, { useEffect, useMemo, useState } from "react";

/** API base resolution */
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
    throw new Error(`GET ${path} → ${res.status} ${res.statusText} ${txt}`);
  }
  return res.json();
}

// NEW: generic JSON POST
async function postJSON(path, body, opts = {}) {
  const res = await fetch(apiURL(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    ...opts,
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`POST ${path} → ${res.status} ${res.statusText} ${txt}`);
  }
  return res.json();
}

export default function App() {
  const API_BASE = useMemo(() => resolvedApiBase, []);
  const [health, setHealth] = useState("checking...");
  const [policy, setPolicy] = useState({
    vector: {},
    reward: 0,
    history: [],
    meta: [],
    metrics: {},
  });
  const [goals, setGoals] = useState([]);
  const [newGoal, setNewGoal] = useState("");
  const [emergence, setEmergence] = useState(null);
  const [error, setError] = useState(null);

  // NEW: chat state
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);
  const [messages, setMessages] = useState([]); // [{role:'user'|'assistant', text, meta?}]
  const [usePerspectives, setUsePerspectives] = useState(false);
  const [model, setModel] = useState(null); // keep null to use backend default

  async function refresh() {
    setError(null);
    try {
      const d = await getJSON("/health");
      setHealth(d.status ?? "ok");
    } catch (e) {
      setHealth("error");
      setError(String(e));
    }

    try {
      const p = await getJSON("/policy");
      setPolicy(p);
    } catch (e) {
      setError((prev) => prev || String(e));
    }

    try {
      const g = await getJSON("/goals");
      setGoals(g.items || []);
    } catch (e) {
      setError((prev) => prev || String(e));
    }

    try {
      const em = await getJSON("/emergence");
      setEmergence(em);
    } catch (e) {
      setError((prev) => prev || String(e));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function addGoal() {
    if (!newGoal.trim()) return;
    setError(null);
    try {
      await postJSON("/goals", { text: newGoal });
      setNewGoal("");
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function activateGoal(id) {
    setError(null);
    try {
      await fetch(apiURL(`/goals/${id}/activate`), { method: "POST" });
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  async function tick() {
    setError(null);
    try {
      await fetch(apiURL("/tick"), { method: "POST" });
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  // NEW: chat handlers
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
        { role: "assistant", text: resp.reply, meta: resp.meta },
      ]);
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", text: `⚠️ ${String(err)}` }]);
    } finally {
      setChatSending(false);
    }
  }

  function onChatKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      sendChat(e);
    }
  }

  return (
    <div className="container">
      <div className="title">🧬 DoubleHelix v0.8.1</div>
      <div className="subtitle">
        Emergence: consolidation • curiosity • multi-scale • prompt evolution • perspectives
      </div>

      {/* Debug row */}
      <div className="row" style={{ marginBottom: 8 }}>
        <span className="pill">API: {API_BASE}</span>
        <span className="pill">health: {health}</span>
        {error && <span className="pill" style={{ background: "#522" }}>err</span>}
        <button className="pill" onClick={refresh}>Refresh</button>
        <button className="pill" onClick={tick}>Planner Tick</button>
      </div>
      {error && (
        <div className="card" style={{ borderColor: "#a44" }}>
          <strong>Error</strong>
          <pre style={{ whiteSpace: "pre-wrap" }}>{error}</pre>
        </div>
      )}

      <div className="grid">
        {/* Policy */}
        <div className="card">
          <h3>Policy</h3>
          <div className="row">
            <span className="pill">reward: {policy.reward}</span>
            <span className="pill">history: {policy.history?.length || 0}</span>
          </div>
          <pre>{JSON.stringify(policy.vector || {}, null, 2)}</pre>
          {policy.metrics && (
            <>
              <h3 style={{ marginTop: 12 }}>Metrics</h3>
              <pre>{JSON.stringify(policy.metrics, null, 2)}</pre>
            </>
          )}
        </div>

        {/* Goals */}
        <div className="card">
          <h3>Goals</h3>
          <div className="row">
            <input
              placeholder="New goal…"
              value={newGoal}
              onChange={(e) => setNewGoal(e.target.value)}
            />
            <button onClick={addGoal}>Add</button>
          </div>
          <div style={{ marginTop: 10 }}>
            {goals.map((g) => (
              <div key={g.id} className="row" style={{ marginBottom: 6 }}>
                <span className="pill">#{g.id}</span>
                <span>{g.text}</span>
                <span className="pill">active: {String(g.active)}</span>
                {!g.active && (
                  <button className="ghost" onClick={() => activateGoal(g.id)}>
                    Activate
                  </button>
                )}
              </div>
            ))}
            {!goals.length && <div className="muted">No goals yet. Add one above.</div>}
          </div>
        </div>

        {/* Emergence */}
        <div className="card">
          <h3>Emergence</h3>
          <pre>{emergence ? JSON.stringify(emergence, null, 2) : "(no data yet)"}</pre>
          <div className="muted">
            This aggregates meta signals like emergent principles, policy stability, and scores.
          </div>
        </div>

        {/* NEW: Chat */}
        <div className="card">
          <h3>Chat</h3>

          {/* Controls */}
          <div className="row" style={{ gap: 8, marginBottom: 8 }}>
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
              {/* add others you support */}
            </select>
          </div>

          {/* Transcript */}
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

          {/* Input */}
          <form onSubmit={sendChat} className="row" style={{ marginTop: 8, gap: 8 }}>
            <input
              placeholder="Type a message… (Enter to send)"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  sendChat(e);
                }
              }}
              disabled={chatSending}
              style={{ flex: 1 }}
            />
            <button disabled={chatSending} className="pill" onClick={sendChat}>
              {chatSending ? "Sending…" : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
