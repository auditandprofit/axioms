import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


@dataclass
class DAG:
    """Simple in-memory representation of a DAG."""

    edges: Dict[str, List[str]] = field(default_factory=dict)

    def _has_path(self, start: str, target: str, visited: Optional[set] = None) -> bool:
        """Return True if ``target`` is reachable from ``start``."""
        if start == target:
            return True
        if visited is None:
            visited = set()
        if start in visited:
            return False
        visited.add(start)
        for nxt in self.edges.get(start, []):
            if self._has_path(nxt, target, visited):
                return True
        return False

    def add_edge(self, parent: str, child: str) -> None:
        if self._has_path(child, parent):
            raise ValueError(f"Adding edge {parent!r}->{child!r} would create a cycle")
        self.edges.setdefault(parent, []).append(child)
        self.edges.setdefault(child, [])

    def to_nested(self, roots: List[str]) -> List[Dict[str, Any]]:
        """Convert the internal edge mapping to a nested structure.

        Each node is represented as an object with ``content`` and ``children``
        fields, where ``children`` is a list of nodes in the same format.
        """

        def build(node: str, path: Optional[set] = None) -> Dict[str, Any]:
            if path is None:
                path = set()
            if node in path:
                raise ValueError(f"Cycle detected at node {node!r}")
            children = [
                build(child, path | {node}) for child in self.edges.get(node, [])
            ]
            return {"content": node, "children": children}

        return [build(root) for root in roots]


FUNCTIONS = [
    {
        "name": "stop_expansion",
        "description": "Indicate that the current node has no further children.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "new_edges",
        "description": "Return a list of child nodes to attach to the current node.",
        "parameters": {
            "type": "object",
            "properties": {
                "children": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of child nodes",
                }
            },
            "required": ["children"],
        },
    },
]

SYSTEM_PROMPT = (
    "You expand nodes in a directed acyclic graph. "
    "Use the 'stop_expansion' function when no further ideas are needed. "
    "Use 'new_edges' to suggest new child nodes to explore."
)


async def expand_node(
    client: "AsyncOpenAI", base_input: str, node: Optional[str]
) -> Tuple[str, List[str]]:
    """Ask the model to expand a single node.

    The model always receives the original ``base_input`` along with the
    ``node`` being expanded to mirror the flow:
        system prompt + input text + new obj -> function call
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": base_input},
    ]
    parent = base_input
    if node is not None:
        messages.append({"role": "user", "content": node})
        parent = node

    response = await client.responses.create(
        model="gpt-5",
        input=messages,
        tools=[{**f, "type": "function"} for f in FUNCTIONS],
        tool_choice="auto",
    )

    name = None
    arguments = "{}"
    for item in response.output or []:
        if getattr(item, "type", None) == "function_call":
            name = getattr(item, "name", None)
            arguments = getattr(item, "arguments", "{}")
            break
    payload = json.loads(arguments)

    if name == "new_edges":
        return parent, payload.get("children", [])
    return parent, []


async def build_dag(
    seeds: List[str],
    max_nodes: int = 50,
    max_depth: Optional[int] = None,
    max_fanout: Optional[int] = None,
) -> DAG:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required to run this script")
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dag = DAG()
    queue: List[Tuple[str, Optional[str], int]] = []
    if max_depth is None or max_depth > 0:
        queue = [(seed, None, 0) for seed in seeds]
    seen = set(seeds)

    while queue and len(dag.edges) < max_nodes:
        layer = queue[0][2]
        tasks: List[asyncio.Task] = []
        in_flight = 0
        for base, node, _ in queue:
            tasks.append(asyncio.create_task(expand_node(client, base, node)))
            in_flight += 1
            print(
                f"Layer {layer}: {in_flight} node(s) in flight",
                end="\r",
                flush=True,
                file=sys.stderr,
            )
        results: List[Tuple[str, List[str]]] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            in_flight -= 1
            print(
                f"Layer {layer}: {in_flight} node(s) in flight",
                end="\r",
                flush=True,
                file=sys.stderr,
            )
        print(file=sys.stderr)
        new_queue: List[Tuple[str, Optional[str], int]] = []
        for (base, _, depth), (parent, children) in zip(queue, results):
            if max_fanout is not None:
                children = children[:max_fanout]
            for child in children:
                if child not in seen:
                    seen.add(child)
                    if max_depth is None or depth + 1 < max_depth:
                        new_queue.append((base, child, depth + 1))
                dag.add_edge(parent, child)
        if new_queue:
            next_layer = new_queue[0][2]
            print(
                f"Layer {next_layer} discovered with {len(new_queue)} node(s)",
                file=sys.stderr,
            )
        queue = new_queue
    return dag


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a DAG using OpenAI function calls")
    parser.add_argument("seed", nargs="*", help="Seed node(s) to start expansion")
    parser.add_argument(
        "--seed-file",
        type=argparse.FileType("r"),
        help="Path to a file whose entire contents form a single seed node",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=50, help="Maximum number of nodes to generate"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth to expand (0 disables expansion)",
    )
    parser.add_argument(
        "--max-fanout",
        type=int,
        default=None,
        help="Maximum number of children per node",
    )
    args = parser.parse_args()

    seeds = list(args.seed)
    if args.seed_file:
        with args.seed_file as f:
            content = f.read().strip()
            if content:
                seeds.append(content)

    if not seeds:
        parser.error("No seeds provided. Specify positional seeds or use --seed-file.")

    dag = asyncio.run(
        build_dag(
            seeds,
            args.max_nodes,
            args.max_depth,
            args.max_fanout,
        )
    )
    nested = dag.to_nested(seeds)
    print(json.dumps(nested, indent=2))


if __name__ == "__main__":
    main()
