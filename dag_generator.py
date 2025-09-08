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


def make_functions(max_fanout: Optional[int]) -> List[Dict[str, Any]]:
    functions = [
        {
            "name": "stop_expansion",
            "description": "Indicate that the given node has no further children.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "description": "Node that should not be expanded further",
                    }
                },
                "required": ["node"],
            },
        },
        {
            "name": "new_edges",
            "description": "Return a list of child nodes to attach to the given node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node": {
                        "type": "string",
                        "description": "Parent node being expanded",
                    },
                    "children": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of child nodes",
                    },
                },
                "required": ["node", "children"],
            },
        },
    ]
    if max_fanout is not None:
        functions[1]["parameters"]["properties"]["children"]["maxItems"] = max_fanout
    return functions


def make_system_prompt(max_fanout: Optional[int]) -> str:
    prompt = (
        "You expand nodes in a directed acyclic graph. "
        "Use the 'stop_expansion' function when no further ideas are needed. "
        "Use 'new_edges' to suggest new child nodes to explore. "
        "For each node, call exactly one function: either 'new_edges' or 'stop_expansion'."
    )
    if max_fanout is not None:
        prompt += f" Do not propose more than {max_fanout} child nodes per parent."
    return prompt


async def expand_layer(
    client: "AsyncOpenAI",
    context: str,
    nodes: List[str],
    max_fanout: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Expand all nodes in ``nodes`` using a single batched request.

    Args:
        client: OpenAI client used to make the request.
        context: Conversation context so far.
        nodes: Nodes to expand.
        max_fanout: Maximum number of children the model may return per node.
    """

    messages = [
        {"role": "system", "content": make_system_prompt(max_fanout)},
        {"role": "user", "content": context},
        {"role": "user", "content": "Expand the following nodes:\n" + "\n".join(nodes)},
    ]

    functions = make_functions(max_fanout)

    response = await client.responses.create(
        model="gpt-4o-mini",
        input=messages,
        tools=[{**f, "type": "function"} for f in functions],
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    expansions: Dict[str, List[str]] = {}
    for item in response.output or []:
        if getattr(item, "type", None) == "function_call":
            name = getattr(item, "name", None)
            arguments = getattr(item, "arguments", "{}")
            payload = json.loads(arguments)
            node = payload.get("node")
            if not node:
                continue
            if name == "new_edges":
                expansions[node] = payload.get("children", [])
            elif name == "stop_expansion":
                expansions[node] = []

    return expansions


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

    queue: List[Tuple[str, int]] = []
    if max_depth is None or max_depth > 0:
        queue = [(seed, 0) for seed in seeds]

    seen = set(seeds)
    context_lines = ["Seeds:"] + seeds

    while queue and len(dag.edges) < max_nodes:
        layer = queue[0][1]
        nodes = [node for node, _ in queue]
        print(
            f"Layer {layer}: expanding {len(nodes)} node(s)",
            file=sys.stderr,
        )

        expansions = await expand_layer(
            client, "\n".join(context_lines), nodes, max_fanout
        )

        new_queue: List[Tuple[str, int]] = []
        context_lines.append(f"Layer {layer}:")
        for parent in nodes:
            children = expansions.get(parent, [])
            if max_fanout is not None:
                children = children[:max_fanout]
            context_lines.append(
                f"{parent} -> {', '.join(children) if children else '(none)'}"
            )
            for child in children:
                dag.add_edge(parent, child)
                if child not in seen:
                    seen.add(child)
                    if max_depth is None or layer + 1 < max_depth:
                        new_queue.append((child, layer + 1))

        if new_queue:
            next_layer = new_queue[0][1]
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
