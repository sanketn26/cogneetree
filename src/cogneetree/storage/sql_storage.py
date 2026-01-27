"""SQL-based storage backends (SQLite, Postgres)."""

import json
import sqlite3
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import Session, Activity, Task, ContextItem, ContextCategory

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:
    psycopg2 = None

logger = logging.getLogger(__name__)


class SqlStorageBase(ContextStorageABC):
    """Base class for SQL storage."""
    pass


class SQLiteStorage(ContextStorageABC):
    """SQLite implementation."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None
    
    def _init_db(self):
        """Initialize tables."""
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY, original_ask TEXT, high_level_plan TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS activities (
            activity_id TEXT PRIMARY KEY, session_id TEXT, description TEXT, tags TEXT, 
            mode TEXT, component TEXT, planner_analysis TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, activity_id TEXT, description TEXT, tags TEXT, result TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS context_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, category TEXT, 
            tags TEXT, parent_id TEXT, timestamp TEXT, metadata TEXT)""")
        self.conn.commit()

    def create_session(self, session_id: str, original_ask: str, high_level_plan: str) -> Session:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
                (session_id, original_ask, high_level_plan, datetime.now().isoformat())
            )
        self.current_session_id = session_id
        return Session(session_id, original_ask, high_level_plan)

    def create_activity(self, activity_id: str, session_id: str, description: str, tags: List[str], mode: str, component: str, planner_analysis: str) -> Activity:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO activities VALUES (?, ?, ?, ?, ?, ?, ?)",
                (activity_id, session_id, description, json.dumps(tags), mode, component, planner_analysis)
            )
        self.current_activity_id = activity_id
        return Activity(activity_id, session_id, description, tags, mode, component, planner_analysis)

    def create_task(self, task_id: str, activity_id: str, description: str, tags: List[str]) -> Task:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?, ?)",
                (task_id, activity_id, description, json.dumps(tags), None)
            )
        self.current_task_id = task_id
        return Task(task_id, activity_id, description, tags)

    def complete_task(self, task_id: str, result: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE tasks SET result = ? WHERE task_id = ?", (result, task_id))

    def add_item(self, content: str, category: ContextCategory, tags: List[str], parent_id: Optional[str] = None, embedding: Optional[Any] = None) -> ContextItem:
        with self.conn:
            self.conn.execute(
                "INSERT INTO context_items (content, category, tags, parent_id, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (content, category.value, json.dumps(tags), parent_id, datetime.now().isoformat(), json.dumps({}))
            )
        return ContextItem(content, category, tags, parent_id=parent_id)

    def get_session(self, session_id: str) -> Optional[Session]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = c.fetchone()
        if row:
            return Session(row[0], row[1], row[2], datetime.fromisoformat(row[3]))
        return None

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM activities WHERE activity_id = ?", (activity_id,))
        row = c.fetchone()
        if row:
            return Activity(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5], row[6])
        return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = c.fetchone()
        if row:
            return Task(row[0], row[1], row[2], json.loads(row[3]), row[4])
        return None

    def get_current_session(self) -> Optional[Session]:
        if self.current_session_id:
            return self.get_session(self.current_session_id)
        # Try to find latest
        c = self.conn.cursor()
        c.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT 1")
        row = c.fetchone()
        if row:
            return Session(row[0], row[1], row[2], datetime.fromisoformat(row[3]))
        return None

    def get_current_activity(self) -> Optional[Activity]:
        if self.current_activity_id:
            return self.get_activity(self.current_activity_id)
        return None

    def get_current_task(self) -> Optional[Task]:
        if self.current_task_id:
            return self.get_task(self.current_task_id)
        return None

    def get_items_by_tags(self, tags: List[str]) -> List[ContextItem]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM context_items")
        rows = c.fetchall()
        
        results = []
        for r in rows:
            item_tags = json.loads(r[3])
            if any(t in item_tags for t in tags):
                results.append(ContextItem(
                    content=r[1],
                    category=ContextCategory(r[2]),
                    tags=item_tags,
                    parent_id=r[4],
                    timestamp=datetime.fromisoformat(r[5]),
                    metadata=json.loads(r[6])
                ))
        return results

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM sessions")
            self.conn.execute("DELETE FROM activities")
            self.conn.execute("DELETE FROM tasks")
            self.conn.execute("DELETE FROM context_items")
        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None


class PostgresStorage(ContextStorageABC):
    """Postgres storage implementation."""

    def __init__(self, dsn: str):
        if psycopg2 is None:
            raise ImportError("Postgres support requires: pip install psycopg2")
        self.dsn = dsn
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self._init_db()

        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None

    def _init_db(self):
        with self.conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, original_ask TEXT, high_level_plan TEXT, created_at TIMESTAMP
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS activities (
                    activity_id TEXT PRIMARY KEY, session_id TEXT, description TEXT, tags JSONB, 
                    mode TEXT, component TEXT, planner_analysis TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, activity_id TEXT, description TEXT, tags JSONB, result TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS context_items (
                    id SERIAL PRIMARY KEY, content TEXT, category TEXT, 
                    tags JSONB, parent_id TEXT, timestamp TIMESTAMP, metadata JSONB
                )
            """)

    def create_session(self, session_id: str, original_ask: str, high_level_plan: str) -> Session:
        with self.conn.cursor() as c:
            c.execute(
                "INSERT INTO sessions VALUES (%s, %s, %s, %s) ON CONFLICT (session_id) DO UPDATE SET original_ask=EXCLUDED.original_ask",
                (session_id, original_ask, high_level_plan, datetime.now())
            )
        self.current_session_id = session_id
        return Session(session_id, original_ask, high_level_plan)

    def create_activity(self, activity_id: str, session_id: str, description: str, tags: List[str], mode: str, component: str, planner_analysis: str) -> Activity:
        with self.conn.cursor() as c:
            c.execute(
                "INSERT INTO activities VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (activity_id) DO NOTHING",
                (activity_id, session_id, description, json.dumps(tags), mode, component, planner_analysis)
            )
        self.current_activity_id = activity_id
        return Activity(activity_id, session_id, description, tags, mode, component, planner_analysis)

    def create_task(self, task_id: str, activity_id: str, description: str, tags: List[str]) -> Task:
        with self.conn.cursor() as c:
            c.execute(
                "INSERT INTO tasks VALUES (%s, %s, %s, %s, %s) ON CONFLICT (task_id) DO NOTHING",
                (task_id, activity_id, description, json.dumps(tags), None)
            )
        self.current_task_id = task_id
        return Task(task_id, activity_id, description, tags)

    def complete_task(self, task_id: str, result: str) -> None:
        with self.conn.cursor() as c:
            c.execute("UPDATE tasks SET result = %s WHERE task_id = %s", (result, task_id))

    def add_item(self, content: str, category: ContextCategory, tags: List[str], parent_id: Optional[str] = None, embedding: Optional[Any] = None) -> ContextItem:
        with self.conn.cursor() as c:
            c.execute(
                "INSERT INTO context_items (content, category, tags, parent_id, timestamp, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (content, category.value, json.dumps(tags), parent_id, datetime.now(), json.dumps({}))
            )
        return ContextItem(content, category, tags, parent_id=parent_id)

    def get_session(self, session_id: str) -> Optional[Session]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
            row = c.fetchone()
            if row:
                return Session(row['session_id'], row['original_ask'], row['high_level_plan'], row['created_at'])
        return None

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM activities WHERE activity_id = %s", (activity_id,))
            row = c.fetchone()
            if row:
                return Activity(row['activity_id'], row['session_id'], row['description'], row['tags'], row['mode'], row['component'], row['planner_analysis'])
        return None

    def get_task(self, task_id: str) -> Optional[Task]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM tasks WHERE task_id = %s", (task_id,))
            row = c.fetchone()
            if row:
                return Task(row['task_id'], row['activity_id'], row['description'], row['tags'], row['result'])
        return None

    def get_current_session(self) -> Optional[Session]:
        if self.current_session_id:
            return self.get_session(self.current_session_id)
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT 1")
            row = c.fetchone()
            if row: return Session(row['session_id'], row['original_ask'], row['high_level_plan'], row['created_at'])
        return None

    def get_current_activity(self) -> Optional[Activity]:
        if self.current_activity_id: return self.get_activity(self.current_activity_id)
        return None

    def get_current_task(self) -> Optional[Task]:
        if self.current_task_id: return self.get_task(self.current_task_id)
        return None

    def get_items_by_tags(self, tags: List[str]) -> List[ContextItem]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM context_items WHERE tags ?| %s", (tags,))
            rows = c.fetchall()
            return [ContextItem(
                r['content'], ContextCategory(r['category']), r['tags'], 
                r['parent_id'], r['timestamp'], r['metadata']
            ) for r in rows]

    def clear(self) -> None:
        with self.conn.cursor() as c:
            c.execute("TRUNCATE sessions, activities, tasks, context_items")
        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None
