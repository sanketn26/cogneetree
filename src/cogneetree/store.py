"""File-backed Markdown decision store."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cogneetree.protocol import DecisionResolution, utc_now


class DecisionFileStore:
    """Persist accepted decisions as Markdown and events as JSONL."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.decisions_dir = self.root / "decisions"
        self.events_dir = self.root / "events"
        self.rejected_dir = self.root / "audit" / "rejected"
        self.events_path = self.events_dir / "decisions.jsonl"

    def initialize(self) -> None:
        self.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

    def decision_path(self, area: str) -> Path:
        parts = [part.strip() for part in area.split("/") if part.strip()]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"Invalid decision area: {area!r}")
        return self.decisions_dir.joinpath(*parts).with_suffix(".md")

    def read_decision(self, area: str) -> str | None:
        path = self.decision_path(area)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def write_decision(self, area: str, markdown: str) -> None:
        path = self.decision_path(area)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(markdown, encoding="utf-8")
        temp_path.replace(path)

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events_dir.mkdir(parents=True, exist_ok=True)
        event = {"type": event_type, "timestamp": utc_now(), "payload": payload}
        with self.events_path.open("a", encoding="utf-8") as events:
            events.write(json.dumps(event, sort_keys=True))
            events.write("\n")

    def write_rejection(self, resolution: DecisionResolution) -> None:
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        path = self.rejected_dir / f"{resolution.proposal_id}.json"
        path.write_text(json.dumps(asdict(resolution), indent=2, sort_keys=True), encoding="utf-8")

    def list_areas(self) -> list[str]:
        if not self.decisions_dir.exists():
            return []
        areas = []
        for path in self.decisions_dir.rglob("*.md"):
            areas.append(path.relative_to(self.decisions_dir).with_suffix("").as_posix())
        return sorted(areas)

