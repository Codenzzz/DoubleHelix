# backend/utils/db.py

import os
import json
import time
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, LargeBinary, Boolean,
    select, update, func
)
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

# ---------------------------------------------------------------------
# Engine / Session
# ---------------------------------------------------------------------
DB_PATH = os.getenv("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "..", "doublehelix.db"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# Add psycopg2 driver if a bare Postgres URL is provided
if DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# For SQLite, enable multithreaded access (FastAPI + background threads)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite:///") else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
Base = declarative_base()


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _norm_text(s: str) -> str:
    """Normalize text for dedupe: collapse spaces + lowercase."""
    return " ".join(str(s or "").split()).lower()


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class Fact(Base):
    __tablename__ = "facts"
    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=0.8)
    ts = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=True)


class Policy(Base):
    __tablename__ = "policy"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(String, nullable=False)
    vector = Column(Text, nullable=False)


class KV(Base):
    __tablename__ = "kv"
    key = Column(String, primary_key=True)
    data = Column(Text, nullable=False)
    ts = Column(String, nullable=False)


class Goal(Base):
    __tablename__ = "goals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    active = Column(Boolean, nullable=False, default=False)
    ts = Column(String, nullable=False)


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------
def init():
    if DATABASE_URL.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------
def upsert_fact(key: str, value: str, conf: float = 0.8, embedding: Optional[bytes] = None):
    ts = _now_str()
    s = SessionLocal()
    try:
        obj = s.get(Fact, key)
        if obj is None:
            obj = Fact(key=key, value=value, confidence=conf, ts=ts, embedding=embedding)
            s.add(obj)
        else:
            obj.value = value
            obj.confidence = conf
            obj.ts = ts
            if embedding is not None:
                obj.embedding = embedding
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def reindex_fts():
    return None


def get_fact(key: str) -> Optional[Dict[str, Any]]:
    s = SessionLocal()
    try:
        obj = s.get(Fact, key)
        if not obj:
            return None
        return {"key": obj.key, "value": obj.value, "confidence": obj.confidence, "ts": obj.ts}
    finally:
        s.close()


def all_facts() -> List[Dict[str, Any]]:
    s = SessionLocal()
    try:
        rows = s.execute(select(Fact).order_by(Fact.ts.desc())).scalars().all()
        return [{"key": r.key, "value": r.value, "confidence": r.confidence, "ts": r.ts} for r in rows]
    finally:
        s.close()


def top_facts(limit: int = 5) -> List[Dict[str, Any]]:
    s = SessionLocal()
    try:
        rows = s.execute(select(Fact).order_by(Fact.ts.desc()).limit(limit)).scalars().all()
        return [{"key": r.key, "value": r.value, "confidence": r.confidence, "ts": r.ts} for r in rows]
    finally:
        s.close()


def search_facts(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Portable search for both Postgres and SQLite.
    Matches in value OR key (useful for 'principle:' etc.).
    """
    s = SessionLocal()
    try:
        pattern = f"%{query}%"
        if DATABASE_URL.startswith("postgresql"):
            filt = (Fact.value.ilike(pattern) | Fact.key.ilike(pattern))
        else:
            filt = (Fact.value.like(pattern) | Fact.key.like(pattern))
        rows = s.execute(select(Fact).where(filt).order_by(Fact.ts.desc()).limit(limit)).scalars().all()
        return [{"key": r.key, "value": r.value, "confidence": r.confidence, "ts": r.ts} for r in rows]
    finally:
        s.close()


# ---------------------------------------------------------------------
# Policy vector
# ---------------------------------------------------------------------
def get_policy_vector(default: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    s = SessionLocal()
    try:
        row = s.execute(select(Policy).order_by(Policy.id.desc()).limit(1)).scalars().first()
        if row:
            try:
                return json.loads(row.vector)
            except Exception:
                pass
        return default or {"creativity": 0.0, "conciseness": 0.0, "skepticism": 0.0, "planning_focus": 0.0}
    finally:
        s.close()


def set_policy_vector(vec: Dict[str, float]):
    s = SessionLocal()
    try:
        s.add(Policy(ts=_now_str(), vector=json.dumps(vec)))
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_policy_history(last: int = 20) -> List[Dict[str, Any]]:
    s = SessionLocal()
    try:
        rows = s.execute(select(Policy).order_by(Policy.id.desc()).limit(last)).scalars().all()
        hist: List[Dict[str, Any]] = []
        for r in rows:
            try:
                hist.append(json.loads(r.vector))
            except Exception:
                hist.append({})
        return hist[::-1]
    finally:
        s.close()


# ---------------------------------------------------------------------
# KV store
# ---------------------------------------------------------------------
def kv_upsert(key: str, data: Dict[str, Any]):
    s = SessionLocal()
    try:
        ts = _now_str()
        obj = s.get(KV, key)
        payload = json.dumps(data)
        if obj is None:
            s.add(KV(key=key, data=payload, ts=ts))
        else:
            obj.data = payload
            obj.ts = ts
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def kv_get(key: str) -> Optional[Dict[str, Any]]:
    s = SessionLocal()
    try:
        obj = s.get(KV, key)
        if not obj:
            return None
        try:
            return json.loads(obj.data)
        except Exception:
            return None
    finally:
        s.close()


# ---------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------
def goals_list() -> List[Dict[str, Any]]:
    s = SessionLocal()
    try:
        rows = s.execute(select(Goal).order_by(Goal.id.desc())).scalars().all()
        return [{"id": g.id, "text": g.text, "active": bool(g.active), "ts": g.ts} for g in rows]
    finally:
        s.close()


def goals_add(text: str) -> int:
    s = SessionLocal()
    try:
        g = Goal(text=text, active=False, ts=_now_str())
        s.add(g)
        s.commit()
        return g.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def goals_activate(goal_id: int):
    s = SessionLocal()
    try:
        s.execute(update(Goal).values(active=False))
        s.execute(update(Goal).where(Goal.id == goal_id).values(active=True))
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def goal_active() -> Optional[Dict[str, Any]]:
    s = SessionLocal()
    try:
        row = s.execute(select(Goal).where(Goal.active.is_(True)).order_by(Goal.id.desc()).limit(1)).scalars().first()
        if row:
            return {"id": row.id, "text": row.text, "ts": row.ts}
        return None
    finally:
        s.close()


def goals_delete(goal_id: int) -> bool:
    """
    Delete a goal by id. Returns True if a row was removed.
    """
    s = SessionLocal()
    try:
        obj = s.get(Goal, goal_id)
        if not obj:
            return False
        s.delete(obj)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def goals_dedupe() -> int:
    """
    Remove duplicate goals by normalized text, keeping the oldest entry.
    Returns the number of rows deleted.
    """
    s = SessionLocal()
    try:
        rows = s.execute(select(Goal).order_by(Goal.id.asc())).scalars().all()  # oldest first
        seen: Dict[str, int] = {}
        to_delete: List[int] = []
        for g in rows:
            key = _norm_text(g.text)
            if key in seen:
                to_delete.append(g.id)
            else:
                seen[key] = g.id

        removed = 0
        if to_delete:
            for gid in to_delete:
                obj = s.get(Goal, gid)
                if obj:
                    s.delete(obj)
                    removed += 1
            s.commit()
        return removed
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
