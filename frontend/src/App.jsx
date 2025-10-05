
import React, { useEffect, useState } from "react";
export default function App() {
  const [health, setHealth] = useState("checking...");
  const [policy, setPolicy] = useState({ vector:{}, reward: 0, history:[], meta:[], metrics:{} });
  const [goals, setGoals] = useState([]);
  const [newGoal, setNewGoal] = useState("");
  const [emergence, setEmergence] = useState(null);

  async function refresh() {
    try { const d = await fetch("/api/health").then(r=>r.json()); setHealth(d.status); } catch {}
    try { const p = await fetch("/api/policy").then(r=>r.json()); setPolicy(p); } catch {}
    try { const g = await fetch("/api/goals").then(r=>r.json()); setGoals(g.items || []); } catch {}
    try { const e = await fetch("/api/emergence").then(r=>r.json()); setEmergence(e); } catch {}
  }
  useEffect(()=>{ refresh(); }, []);

  async function addGoal(){
    if(!newGoal.trim()) return;
    await fetch("/api/goals",{method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ text: newGoal })});
    setNewGoal(""); refresh();
  }
  async function activateGoal(id){ await fetch(`/api/goals/${id}/activate`,{method:"POST"}); refresh(); }
  async function tick(){ await fetch("/api/tick",{method:"POST"}); refresh(); }

  return (<div className="container">
    <div className="title">🧬 DoubleHelix v0.8.1</div>
    <div className="subtitle">Emergence: consolidation • curiosity • multi-scale • prompt evolution • perspectives</div>

    <div className="grid">
      <div className="card">
        <h3>System</h3>
        <div className="row"><span className="pill">health: {health}</span>
        <button className="pill" onClick={refresh}>Refresh</button>
        <button className="pill" onClick={tick}>Planner Tick</button></div>
      </div>

      <div className="card">
        <h3>Policy</h3>
        <div className="row">
          <span className="pill">reward: {policy.reward}</span>
          <span className="pill">history: {policy.history?.length || 0}</span>
        </div>
        <pre>{JSON.stringify(policy.vector || {}, null, 2)}</pre>
        {policy.metrics && <>
          <h3 style={{marginTop:12}}>Metrics</h3>
          <pre>{JSON.stringify(policy.metrics, null, 2)}</pre>
        </>}
      </div>

      <div className="card">
        <h3>Goals</h3>
        <div className="row">
          <input placeholder="New goal…" value={newGoal} onChange={e=>setNewGoal(e.target.value)} />
          <button onClick={addGoal}>Add</button>
        </div>
        <div style={{marginTop:10}}>
          {goals.map(g => (
            <div key={g.id} className="row" style={{marginBottom:6}}>
              <span className="pill">#{g.id}</span>
              <span>{g.text}</span>
              <span className="pill">active: {String(g.active)}</span>
              {!g.active && <button className="ghost" onClick={()=>activateGoal(g.id)}>Activate</button>}
            </div>
          ))}
          {!goals.length && <div className="muted">No goals yet. Add one above.</div>}
        </div>
      </div>

      <div className="card">
        <h3>Emergence</h3>
        <pre>{emergence ? JSON.stringify(emergence, null, 2) : "(no data yet)"}</pre>
        <div className="muted">This aggregates meta signals like emergent principles, policy stability, and scores.</div>
      </div>
    </div>
  </div>);
}
