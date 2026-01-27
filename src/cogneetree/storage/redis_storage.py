"""Redis storage backend."""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from cogneetree.core.interfaces import ContextStorageABC
from cogneetree.core.models import Session, Activity, Task, ContextItem, ContextCategory

try:
    import redis
except ImportError:
    redis = None


class RedisStorage(ContextStorageABC):
    """Redis storage implementation."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        if redis is None:
            raise ImportError("Redis support requires: pip install redis")
        
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.prefix = "cogneetree:"
        
        self.current_session_key = f"{self.prefix}current:session"
        self.current_activity_key = f"{self.prefix}current:activity"
        self.current_task_key = f"{self.prefix}current:task"

    def _key(self, type_: str, id_: str) -> str:
        return f"{self.prefix}{type_}:{id_}"

    def create_session(self, session_id: str, original_ask: str, high_level_plan: str) -> Session:
        session = Session(session_id, original_ask, high_level_plan)
        data = {
            "session_id": session_id,
            "original_ask": original_ask,
            "high_level_plan": high_level_plan,
            "created_at": session.created_at.isoformat()
        }
        key = self._key("session", session_id)
        self.client.hset(key, mapping=data)
        self.client.set(self.current_session_key, session_id)
        return session

    def create_activity(self, activity_id: str, session_id: str, description: str, tags: List[str], mode: str, component: str, planner_analysis: str) -> Activity:
        activity = Activity(activity_id, session_id, description, tags, mode, component, planner_analysis)
        data = {
            "activity_id": activity_id,
            "session_id": session_id,
            "description": description,
            "tags": json.dumps(tags),
            "mode": mode,
            "component": component,
            "planner_analysis": planner_analysis
        }
        key = self._key("activity", activity_id)
        self.client.hset(key, mapping=data)
        self.client.set(self.current_activity_key, activity_id)
        return activity

    def create_task(self, task_id: str, activity_id: str, description: str, tags: List[str]) -> Task:
        task = Task(task_id, activity_id, description, tags)
        data = {
            "task_id": task_id,
            "activity_id": activity_id,
            "description": description,
            "tags": json.dumps(tags),
            "result": ""
        }
        key = self._key("task", task_id)
        self.client.hset(key, mapping=data)
        self.client.set(self.current_task_key, task_id)
        return task

    def complete_task(self, task_id: str, result: str) -> None:
        key = self._key("task", task_id)
        if self.client.exists(key):
            self.client.hset(key, "result", result)

    def add_item(self, content: str, category: ContextCategory, tags: List[str], parent_id: Optional[str] = None, embedding: Optional[Any] = None) -> ContextItem:
        item = ContextItem(content, category, tags, parent_id=parent_id)
        item_data = {
            "content": content,
            "category": category.value,
            "tags": json.dumps(tags),
            "parent_id": parent_id or "",
            "timestamp": item.timestamp.isoformat(),
            "metadata": json.dumps(item.metadata)
        }
        # Use a list for items, or incrementing ID. 
        # For simplicity, using a global list + hash might be better, but list is OK for simple logs
        # But we need to query by tags.
        # Let's use an incrementing ID for items
        item_id = self.client.incr(f"{self.prefix}ids:items")
        key = self._key("item", str(item_id))
        self.client.hset(key, mapping=item_data)
        
        # Index tags
        for tag in tags:
            self.client.sadd(f"{self.prefix}tag:{tag}", key)
            
        return item

    def get_session(self, session_id: str) -> Optional[Session]:
        key = self._key("session", session_id)
        data = self.client.hgetall(key)
        if not data: return None
        return Session(
            data["session_id"], data["original_ask"], data["high_level_plan"], 
            created_at=datetime.fromisoformat(data["created_at"])
        )

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        key = self._key("activity", activity_id)
        data = self.client.hgetall(key)
        if not data: return None
        return Activity(
            data["activity_id"], data["session_id"], data["description"], 
            json.loads(data["tags"]), data["mode"], data["component"], data["planner_analysis"]
        )

    def get_task(self, task_id: str) -> Optional[Task]:
        key = self._key("task", task_id)
        data = self.client.hgetall(key)
        if not data: return None
        return Task(
            data["task_id"], data["activity_id"], data["description"], 
            json.loads(data["tags"]), result=data.get("result")
        )

    def get_current_session(self) -> Optional[Session]:
        sid = self.client.get(self.current_session_key)
        return self.get_session(sid) if sid else None

    def get_current_activity(self) -> Optional[Activity]:
        aid = self.client.get(self.current_activity_key)
        return self.get_activity(aid) if aid else None

    def get_current_task(self) -> Optional[Task]:
        tid = self.client.get(self.current_task_key)
        return self.get_task(tid) if tid else None

    def get_items_by_tags(self, tags: List[str]) -> List[ContextItem]:
        item_keys = set()
        for tag in tags:
            keys = self.client.smembers(f"{self.prefix}tag:{tag}")
            item_keys.update(keys)
        
        results = []
        for key in item_keys:
            data = self.client.hgetall(key)
            if data:
                results.append(ContextItem(
                    content=data["content"],
                    category=ContextCategory(data["category"]),
                    tags=json.loads(data["tags"]),
                    parent_id=data["parent_id"] or None,
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    metadata=json.loads(data["metadata"])
                ))
        return results

    def clear(self) -> None:
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            self.client.delete(*keys)
