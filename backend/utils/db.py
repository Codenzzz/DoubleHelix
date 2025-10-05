import os, time, sqlite3, json
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "doublehelix.db")

def _conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init():
    with _conn() as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.8,
                ts TEXT NOT NULL,
                embedding BLOB
            )
        """)
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(key, value)")
        con.execute("""
            CREATE TABLE IF NOT EXISTS policy (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                vector TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                ts TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 0,
                ts TEXT NOT NULL
            )
        """)
        con.commit()

# --- FIXED: FTS-safe upsert (no ON CONFLICT on virtual table) ---
def upsert_fact(key: str, value: str, conf: float = 0.8, embedding: bytes | None = None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as con:
        # upsert into main table
        con.execute("""
            INSERT INTO facts(key,value,confidence,ts,embedding)
            VALUES(?,?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                confidence=excluded.confidence,
                ts=excluded.ts,
                embedding=COALESCE(excluded.embedding, facts.embedding)
        """, (key, value, conf, ts, embedding))

        # safely refresh the FTS entry
        row = con.execute("SELECT rowid FROM facts WHERE key=?", (key,)).fetchone()
        if row:
            rid = row[0]
            con.execute("DELETE FROM facts_fts WHERE rowid=?", (rid,))
            con.execute("INSERT INTO facts_fts(rowid, key, value) VALUES(?,?,?)", (rid, key, value))
        con.commit()

# --- NEW: helper to rebuild the FTS index if needed ---
def reindex_fts():
    """Rebuild the FTS index from facts table."""
    with _conn() as con:
        con.execute("DELETE FROM facts_fts")
        con.execute("INSERT INTO facts_fts(rowid, key, value) SELECT rowid, key, value FROM facts")
        con.commit()

def get_fact(key: str) -> Dict[str, Any] | None:
    with _conn() as con:
        cur = con.execute("SELECT key,value,confidence,ts FROM facts WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            k,v,c,ts = row
            return {"key":k,"value":v,"confidence":c,"ts":ts}
    return None

def all_facts() -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute("SELECT key,value,confidence,ts FROM facts ORDER BY ts DESC")
        return [{"key":k,"value":v,"confidence":c,"ts":ts} for (k,v,c,ts) in cur.fetchall()]

def top_facts(limit: int = 5) -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute("SELECT key,value,confidence,ts FROM facts ORDER BY ts DESC LIMIT ?", (limit,))
        return [{"key":k,"value":v,"confidence":c,"ts":ts} for (k,v,c,ts) in cur.fetchall()]

def search_facts(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    tokens = " ".join(t for t in query.split() if len(t) > 2)
    with _conn() as con:
        try:
            cur = con.execute("""
                SELECT f.key, f.value, f.confidence, f.ts
                FROM facts_fts s JOIN facts f ON f.rowid = s.rowid
                WHERE s.value MATCH ? LIMIT ?
            """, (tokens, limit))
            return [{"key":k,"value":v,"confidence":c,"ts":ts} for (k,v,c,ts) in cur.fetchall()]
        except sqlite3.OperationalError:
            cur = con.execute("SELECT key,value,confidence,ts FROM facts WHERE value LIKE ? ORDER BY ts DESC LIMIT ?", (f"%{query}%", limit))
            return [{"key":k,"value":v,"confidence":c,"ts":ts} for (k,v,c,ts) in cur.fetchall()]

# --- Policy vector ---
def get_policy_vector(default: Dict[str, float] | None = None) -> Dict[str, float]:
    with _conn() as con:
        cur = con.execute("SELECT vector FROM policy ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                pass
    return default or {"creativity":0.0, "conciseness":0.0, "skepticism":0.0, "planning_focus":0.0}

def set_policy_vector(vec: Dict[str, float]):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as con:
        con.execute("INSERT INTO policy(ts, vector) VALUES(?,?)", (ts, json.dumps(vec)))
        con.commit()

def get_policy_history(last: int = 20) -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute("SELECT ts, vector FROM policy ORDER BY id DESC LIMIT ?", (last,))
        rows = cur.fetchall()
        hist = []
        for (ts, vec) in rows:
            try:
                hist.append(json.loads(vec))
            except:
                hist.append({})
        return hist[::-1]

# --- KV store ---
def kv_upsert(key: str, data: Dict[str, Any]):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as con:
        con.execute("""
            INSERT INTO kv(key, data, ts)
            VALUES(?,?,?)
            ON CONFLICT(key) DO UPDATE SET
                data=excluded.data,
                ts=excluded.ts
        """, (key, json.dumps(data), ts))
        con.commit()

def kv_get(key: str) -> Dict[str, Any] | None:
    with _conn() as con:
        cur = con.execute("SELECT data FROM kv WHERE key=?", (key,))
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return None
    return None

# --- Goals ---
def goals_list() -> List[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute("SELECT id, text, active, ts FROM goals ORDER BY id DESC")
        return [{"id":i, "text":t, "active":bool(a), "ts":ts} for (i,t,a,ts) in cur.fetchall()]

def goals_add(text: str) -> int:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with _conn() as con:
        cur = con.execute("INSERT INTO goals(text, active, ts) VALUES(?,?,?)", (text, 0, ts))
        con.commit()
        return cur.lastrowid

def goals_activate(goal_id: int):
    with _conn() as con:
        con.execute("UPDATE goals SET active=0")
        con.execute("UPDATE goals SET active=1 WHERE id=?", (goal_id,))
        con.commit()

def goal_active() -> Dict[str, Any] | None:
    with _conn() as con:
        cur = con.execute("SELECT id, text, ts FROM goals WHERE active=1 ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row:
            i,t,ts = row
            return {"id":i, "text":t, "ts":ts}
    return None
