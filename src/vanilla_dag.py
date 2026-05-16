from __future__ import annotations
from collections import deque
from enum import Enum
from typing import Callable


class Status(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED  = "FAILED"
    BLOCKED = "BLOCKED"


class Node:
    def __init__(self, name: str, func: Callable):
        self.name       = name
        self.func       = func
        self.status     = Status.PENDING
        self.upstream:   list[Node] = []
        self.downstream: list[Node] = []
        self.error:      Exception | None = None


class DAG:
    def __init__(self, name: str = "dag"):
        self.name    = name
        self.nodes:   dict[str, Node] = {}
        self.context: dict            = {}  # shared store passed to every node

    def add_node(self, name: str, func: Callable) -> Node:
        node = Node(name, func)
        self.nodes[name] = node
        return node

    def add_edge(self, upstream_name: str, downstream_name: str) -> None:
        up   = self.nodes[upstream_name]
        down = self.nodes[downstream_name]
        up.downstream.append(down)
        down.upstream.append(up)

    # Kahn's algorithm: BFS-based topological sort using in-degree tracking.
    # Guarantees that every node is visited only after all its dependencies.
    def _topological_sort(self) -> list[Node]:
        in_degree = {node: len(node.upstream) for node in self.nodes.values()}
        queue     = deque(node for node, deg in in_degree.items() if deg == 0)
        ordered: list[Node] = []

        while queue:
            node = queue.popleft()
            ordered.append(node)
            for child in node.downstream:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(self.nodes):
            raise ValueError(f"Cycle detected in DAG '{self.name}'")

        return ordered

    # BFS from the failed node — marks all descendants as BLOCKED.
    def _propagate_failure(self, failed_node: Node) -> None:
        queue = deque(failed_node.downstream)
        while queue:
            node = queue.popleft()
            if node.status not in (Status.SUCCESS, Status.FAILED):
                node.status = Status.BLOCKED
                print(f"  [BLOCKED]  {node.name}")
            queue.extend(node.downstream)

    def _execute(self, order: list[Node]) -> None:
        for node in order:
            if node.status == Status.BLOCKED:
                continue

            blocked_by = [u for u in node.upstream if u.status != Status.SUCCESS]
            if blocked_by:
                node.status = Status.BLOCKED
                print(f"  [BLOCKED]  {node.name}")
                self._propagate_failure(node)
                continue

            node.status = Status.RUNNING
            print(f"  [RUNNING]  {node.name}")

            try:
                node.func(self.context)
                node.status = Status.SUCCESS
                print(f"  [SUCCESS]  {node.name}")
            except Exception as exc:
                node.status = Status.FAILED
                node.error  = exc
                print(f"  [FAILED]   {node.name} — {exc}")
                self._propagate_failure(node)

    def run(self) -> None:
        print(f"\n=== {self.name} ===\n")
        order = self._topological_sort()
        self._execute(order)
        self._print_report(order)

    def reprocess(self, *node_names: str) -> None:
        """Reset a node and all its transitive upstream dependencies, then rerun them.

        Nodes outside the reprocess subgraph keep their existing status — SUCCESS
        results from a previous run are reused as-is, so only the affected slice
        of the pipeline is re-executed.
        """
        # BFS upstream to collect every ancestor of the target nodes
        to_reset: set[Node] = set()
        frontier = deque(self.nodes[n] for n in node_names)
        while frontier:
            node = frontier.popleft()
            if node in to_reset:
                continue
            to_reset.add(node)
            frontier.extend(node.upstream)

        for node in to_reset:
            node.status = Status.PENDING
            node.error  = None

        full_order      = self._topological_sort()
        reprocess_order = [n for n in full_order if n in to_reset]

        names = ", ".join(node_names)
        print(f"\n=== {self.name} [reprocess: {names}] ===")
        print(f"    resetting {len(to_reset)} node(s): {[n.name for n in reprocess_order]}\n")

        self._execute(reprocess_order)
        self._print_report(full_order)

    def _print_report(self, order: list[Node]) -> None:
        icons = {
            Status.SUCCESS: "✓",
            Status.FAILED:  "✗",
            Status.BLOCKED: "🔒",
            Status.PENDING: "·",
            Status.RUNNING: "→",
        }
        width = max(len(n.name) for n in order) + 2
        print(f"\n{'━' * (width + 18)}")
        for node in order:
            icon  = icons[node.status]
            label = node.name.ljust(width)
            print(f"  {label} {icon} {node.status.value}")
        print(f"{'━' * (width + 18)}\n")
