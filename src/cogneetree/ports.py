"""Storage ports for context memory."""

from __future__ import annotations

from typing import Any, Protocol

from cogneetree.protocol import DecisionResolution


class DecisionStore(Protocol):
    """Storage needed by the decision admission protocol."""

    def initialize(self) -> None:
        """Prepare storage for reads and writes."""

    def read_decision(self, area: str) -> str | None:
        """Return accepted Markdown for an area."""

    def write_decision(self, area: str, markdown: str) -> None:
        """Persist accepted Markdown for an area."""

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append an audit event."""

    def write_rejection(self, resolution: DecisionResolution) -> None:
        """Persist a rejected proposal audit record."""

    def list_areas(self) -> list[str]:
        """Return accepted decision areas."""
