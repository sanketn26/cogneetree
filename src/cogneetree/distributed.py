"""Distributed context memory API boundaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from cogneetree.leader import MemoryLeader
from cogneetree.memory import ProposalInput
from cogneetree.ports import DecisionStore
from cogneetree.protocol import DecisionProposal, DecisionResolution


@dataclass(frozen=True)
class MemoryCommand:
    """Command sent to the active memory leader."""

    command_id: str
    command_type: str
    payload: dict
    submitted_by: str


class CommandSubmitter(Protocol):
    """Client-side port for sending memory commands."""

    def submit(self, command: MemoryCommand) -> DecisionResolution:
        """Submit a command and return the leader resolution."""


class DecisionReader(Protocol):
    """Client-side port for reading accepted memory."""

    def get_decision(self, area: str) -> str | None:
        """Return accepted Markdown for an area."""

    def list_decisions(self) -> list[str]:
        """Return accepted decision areas."""


class DistributedContextMemory:
    """Agent-facing API for distributed context memory."""

    def __init__(self, submitter: CommandSubmitter, reader: DecisionReader) -> None:
        self.submitter = submitter
        self.reader = reader

    def propose_decision(self, proposal_input: ProposalInput) -> DecisionResolution:
        proposal = DecisionProposal(
            area=proposal_input.area,
            content=proposal_input.content,
            rationale=proposal_input.rationale,
            agent_id=proposal_input.agent_id,
            evidence=proposal_input.evidence,
        )
        command = MemoryCommand(
            command_id=proposal.proposal_id,
            command_type="propose_decision",
            payload=asdict(proposal),
            submitted_by=proposal.agent_id,
        )
        return self.submitter.submit(command)

    def get_decision(self, area: str) -> str | None:
        return self.reader.get_decision(area)

    def list_decisions(self) -> list[str]:
        return self.reader.list_decisions()


class LeaderCommandHandler:
    """Apply memory commands through the protocol leader."""

    def __init__(self, leader_id: str, store: DecisionStore) -> None:
        self.leader = MemoryLeader(leader_id, store)

    def handle(self, command: MemoryCommand) -> DecisionResolution:
        if command.command_type != "propose_decision":
            raise ValueError(f"Unsupported command type: {command.command_type}")
        proposal = DecisionProposal(**command.payload)
        return self.leader.review(proposal)


class LocalCommandSubmitter:
    """In-process submitter for tests and single-node deployments."""

    def __init__(self, handler: LeaderCommandHandler) -> None:
        self.handler = handler

    def submit(self, command: MemoryCommand) -> DecisionResolution:
        return self.handler.handle(command)


class LocalDecisionReader:
    """In-process reader backed by a decision store."""

    def __init__(self, store: DecisionStore) -> None:
        self.store = store

    def get_decision(self, area: str) -> str | None:
        return self.store.read_decision(area)

    def list_decisions(self) -> list[str]:
        return self.store.list_areas()
