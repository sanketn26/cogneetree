"""SQL-based storage backends (SQLite, Postgres)."""

import json
import sqlite3
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import Session, Activity, Task, ContextItem, ContextCategory, ImportanceTier

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

    def add_item(
        self,
        content: str,
        category: ContextCategory,
        tags: List[str],
        parent_id: Optional[str] = None,
        embedding: Optional[Any] = None,
        tier: Optional[Any] = None,
    ) -> ContextItem:
        metadata = {"tier": (tier or ImportanceTier.MINOR).value}
        with self.conn:
            self.conn.execute(
                "INSERT INTO context_items (content, category, tags, parent_id, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (content, category.value, json.dumps(tags), parent_id, datetime.now().isoformat(), json.dumps(metadata))
            )
        return ContextItem(content, category, tags, parent_id=parent_id, metadata=metadata, tier=tier or ImportanceTier.MINOR)

    @staticmethod
    def _row_to_item(row: Tuple[Any, ...]) -> ContextItem:
        metadata = json.loads(row[6]) if row[6] else {}
        tier_value = metadata.get("tier", ImportanceTier.MINOR.value)
        try:
            tier = ImportanceTier(tier_value)
        except ValueError:
            tier = ImportanceTier.MINOR
        return ContextItem(
            content=row[1],
            category=ContextCategory(row[2]),
            tags=json.loads(row[3]),
            parent_id=row[4],
            timestamp=datetime.fromisoformat(row[5]),
            metadata=metadata,
            tier=tier,
        )

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
        return [self._row_to_item(r) for r in rows if any(t in json.loads(r[3]) for t in tags)]

    def get_items_by_category(self, category: ContextCategory) -> List[ContextItem]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM context_items WHERE category = ?", (category.value,))
        return [self._row_to_item(r) for r in c.fetchall()]

    def get_stats(self) -> Dict[str, int]:
        c = self.conn.cursor()
        counts = {}
        for table in ("sessions", "activities", "tasks", "context_items"):
            c.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = c.fetchone()[0]
        return {
            "sessions": counts["sessions"],
            "activities": counts["activities"],
            "tasks": counts["tasks"],
            "items": counts["context_items"],
        }

    def get_items_by_task(self, task_id: str) -> List[ContextItem]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM context_items WHERE parent_id = ?", (task_id,))
        return [self._row_to_item(r) for r in c.fetchall()]

    def get_items_by_activity(self, activity_id: str) -> List[ContextItem]:
        c = self.conn.cursor()
        c.execute("SELECT task_id FROM tasks WHERE activity_id = ?", (activity_id,))
        task_ids = [r[0] for r in c.fetchall()]
        if not task_ids:
            return []
        placeholders = ",".join("?" for _ in task_ids)
        c.execute(f"SELECT * FROM context_items WHERE parent_id IN ({placeholders})", task_ids)
        return [self._row_to_item(r) for r in c.fetchall()]

    def get_items_by_session(self, session_id: str) -> List[ContextItem]:
        c = self.conn.cursor()
        c.execute("SELECT activity_id FROM activities WHERE session_id = ?", (session_id,))
        activity_ids = [r[0] for r in c.fetchall()]
        if not activity_ids:
            return []
        placeholders = ",".join("?" for _ in activity_ids)
        c.execute(f"SELECT task_id FROM tasks WHERE activity_id IN ({placeholders})", activity_ids)
        task_ids = [r[0] for r in c.fetchall()]
        if not task_ids:
            return []
        placeholders = ",".join("?" for _ in task_ids)
        c.execute(f"SELECT * FROM context_items WHERE parent_id IN ({placeholders})", task_ids)
        return [self._row_to_item(r) for r in c.fetchall()]

    def get_all_sessions(self) -> List[Session]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM sessions ORDER BY created_at")
        return [Session(r[0], r[1], r[2], datetime.fromisoformat(r[3])) for r in c.fetchall()]

    @staticmethod
    def _row_to_task(row: Tuple[Any, ...]) -> Task:
        # Schema: task_id, activity_id, description, tags (JSON), result
        return Task(row[0], row[1], row[2], json.loads(row[3]), row[4])

    def get_activity_tasks(self, activity_id: str) -> List[Task]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM tasks WHERE activity_id = ?", (activity_id,))
        return [self._row_to_task(r) for r in c.fetchall()]

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

    def add_item(
        self,
        content: str,
        category: ContextCategory,
        tags: List[str],
        parent_id: Optional[str] = None,
        embedding: Optional[Any] = None,
        tier: Optional[Any] = None,
    ) -> ContextItem:
        metadata = {"tier": (tier or ImportanceTier.MINOR).value}
        with self.conn.cursor() as c:
            c.execute(
                "INSERT INTO context_items (content, category, tags, parent_id, timestamp, metadata) VALUES (%s, %s, %s, %s, %s, %s)",
                (content, category.value, json.dumps(tags), parent_id, datetime.now(), json.dumps(metadata))
            )
        return ContextItem(content, category, tags, parent_id=parent_id, metadata=metadata, tier=tier or ImportanceTier.MINOR)

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
                r['parent_id'], r['timestamp'], r['metadata'],
                tier=ImportanceTier(r.get("metadata", {}).get("tier", ImportanceTier.MINOR.value))
            ) for r in rows]

    def get_items_by_category(self, category: ContextCategory) -> List[ContextItem]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM context_items WHERE category = %s", (category.value,))
            rows = c.fetchall()
            return [ContextItem(
                r['content'], ContextCategory(r['category']), r['tags'],
                r['parent_id'], r['timestamp'], r['metadata'],
                tier=ImportanceTier(r.get("metadata", {}).get("tier", ImportanceTier.MINOR.value))
            ) for r in rows]

    def get_stats(self) -> Dict[str, int]:
        with self.conn.cursor() as c:
            c.execute("SELECT COUNT(*) FROM sessions")
            sessions = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM activities")
            activities = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM tasks")
            tasks = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM context_items")
            items = c.fetchone()[0]
        return {"sessions": sessions, "activities": activities, "tasks": tasks, "items": items}

    def get_items_by_task(self, task_id: str) -> List[ContextItem]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM context_items WHERE parent_id = %s", (task_id,))
            rows = c.fetchall()
            return [ContextItem(
                r['content'], ContextCategory(r['category']), r['tags'],
                r['parent_id'], r['timestamp'], r['metadata'],
                tier=ImportanceTier(r.get("metadata", {}).get("tier", ImportanceTier.MINOR.value))
            ) for r in rows]

    def get_items_by_activity(self, activity_id: str) -> List[ContextItem]:
        with self.conn.cursor() as c:
            c.execute("SELECT task_id FROM tasks WHERE activity_id = %s", (activity_id,))
            task_ids = [r[0] for r in c.fetchall()]
        if not task_ids:
            return []
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM context_items WHERE parent_id = ANY(%s)", (task_ids,))
            rows = c.fetchall()
            return [ContextItem(
                r['content'], ContextCategory(r['category']), r['tags'],
                r['parent_id'], r['timestamp'], r['metadata'],
                tier=ImportanceTier(r.get("metadata", {}).get("tier", ImportanceTier.MINOR.value))
            ) for r in rows]

    def get_items_by_session(self, session_id: str) -> List[ContextItem]:
        with self.conn.cursor() as c:
            c.execute("SELECT activity_id FROM activities WHERE session_id = %s", (session_id,))
            activity_ids = [r[0] for r in c.fetchall()]
            if not activity_ids:
                return []
            c.execute("SELECT task_id FROM tasks WHERE activity_id = ANY(%s)", (activity_ids,))
            task_ids = [r[0] for r in c.fetchall()]
        if not task_ids:
            return []
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM context_items WHERE parent_id = ANY(%s)", (task_ids,))
            rows = c.fetchall()
            return [ContextItem(
                r['content'], ContextCategory(r['category']), r['tags'],
                r['parent_id'], r['timestamp'], r['metadata'],
                tier=ImportanceTier(r.get("metadata", {}).get("tier", ImportanceTier.MINOR.value))
            ) for r in rows]

    def get_all_sessions(self) -> List[Session]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM sessions ORDER BY created_at")
            rows = c.fetchall()
            return [Session(r['session_id'], r['original_ask'], r['high_level_plan'], r['created_at']) for r in rows]

    def get_activity_tasks(self, activity_id: str) -> List[Task]:
        with self.conn.cursor(cursor_factory=DictCursor) as c:
            c.execute("SELECT * FROM tasks WHERE activity_id = %s", (activity_id,))
            rows = c.fetchall()
            return [Task(r['task_id'], r['activity_id'], r['description'], r['tags'], r['result']) for r in rows]

    def clear(self) -> None:
        with self.conn.cursor() as c:
            c.execute("TRUNCATE sessions, activities, tasks, context_items")
        self.current_session_id = None
        self.current_activity_id = None
        self.current_task_id = None
