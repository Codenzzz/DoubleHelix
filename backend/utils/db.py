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
# Prefer Postgres from env; fall back to a persistent SQLite path for local use.
DB_PATH = os.getenv("SQLITE_PATH", os.path.join(os.path.dirname(__file__), "..", "doublehelix.db"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# psycopg2 driver hint if user pasted a bare postgres URL
if DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
Base = declarative_base()


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------
# Models (schema-compatible with your previous tables)
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
# Init (creates tables if missing). FTS table is removed (Postgres-safe).
# ---------------------------------------------------------------------
def init():
    # ensure local dir for SQLite if used
    if DATABASE_URL.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------
def upsert_fact(key: str, value: str, conf: float = 0.8, embedding: Optional[bytes] = None):
    """Insert or update a fact by key (cross-DB safe)."""
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
    """No-op now (used to rebuild SQLite FTS). Kept for compatibility."""
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
    Portable search that works on both Postgres and SQLite.
    - Postgres: ILIKE (case-insensitive)
    - SQLite: LIKE (case-insensitive-ish by default for ASCII)
    """
    s = SessionLocal()
    try:
        pattern = f"%{query}%"
        if DATABASE_URL.startswith("postgresql"):
            filt = Fact.value.ilike(pattern)  # Postgres ILIKE
        else:
            filt = Fact.value.like(pattern)
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
        return hist[::-1]  # oldest first
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
        # Deactivate all, then activate the chosen one.
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
