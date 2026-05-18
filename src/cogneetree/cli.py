"""Minimal command line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from cogneetree.leader import MemoryLeader
from cogneetree.protocol import DecisionProposal, ProposalStatus
from cogneetree.store import DecisionFileStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="cogneetree")
    parser.add_argument("--memory", default="memory", help="Memory directory")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("init")

    propose = subcommands.add_parser("propose-decision")
    propose.add_argument("area")
    propose.add_argument("--content", required=True)
    propose.add_argument("--rationale", required=True)
    propose.add_argument("--agent", required=True)
    propose.add_argument("--leader", default="local-leader")
    propose.add_argument("--evidence", action="append", default=[])

    decisions = subcommands.add_parser("decisions")
    decision_commands = decisions.add_subparsers(dest="decision_command", required=True)
    decision_commands.add_parser("list")
    show = decision_commands.add_parser("show")
    show.add_argument("area")

    args = parser.parse_args()
    store = DecisionFileStore(Path(args.memory))

    if args.command == "init":
        store.initialize()
        print(f"Initialized {store.root}")
        return

    if args.command == "propose-decision":
        proposal = DecisionProposal(
            area=args.area,
            content=args.content,
            rationale=args.rationale,
            agent_id=args.agent,
            evidence=args.evidence,
        )
        resolution = MemoryLeader(args.leader, store).review(proposal)
        if resolution.status == ProposalStatus.ACCEPTED:
            print(f"ACCEPTED {resolution.area} v{resolution.latest_version}")
            return
        print(f"REJECTED_STALE {resolution.area}")
        print()
        print(resolution.reason)
        print("Re-evaluate using the latest accepted state.")
        return

    if args.command == "decisions":
        store.initialize()
        if args.decision_command == "list":
            for area in store.list_areas():
                print(area)
            return
        markdown = store.read_decision(args.area)
        if markdown is None:
            raise SystemExit(f"No decision found for area: {args.area}")
        print(markdown)
        return


if __name__ == "__main__":
    main()

