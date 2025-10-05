import React, { useEffect, useMemo, useState } from "react";

/** API base resolution:
 *  - Use VITE_API_BASE in Render builds
 *  - Use localhost:8000 during local dev
 *  - Else fallback to the live backend
 */
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
      await fetch(apiURL("/goals"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: newGoal }),
      });
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

        <div className="card">
          <h3>Emergence</h3>
          <pre>{emergence ? JSON.stringify(emergence, null, 2) : "(no data yet)"}</pre>
          <div className="muted">
            This aggregates meta signals like emer
