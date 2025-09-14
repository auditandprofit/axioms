import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


_LOG_DIR: Optional[Path] = None
_RESPONSE_COUNT = 0

# Counter used to assign sequential IDs to nodes
_NODE_COUNTER = 0


def _next_id() -> str:
    """Return a new sequential node id prefixed with ``axiom_node_id-``."""
    global _NODE_COUNTER
    node_id = _NODE_COUNTER
    _NODE_COUNTER += 1
    return f"axiom_node_id-{node_id}"


@dataclass
class Node:
    """Small helper structure for nodes identified by an id and text."""

    id: str
    text: str


def _store_response(response: Any) -> None:
    """Persist function call and text data from a response to disk."""

    global _LOG_DIR, _RESPONSE_COUNT
    if _LOG_DIR is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _LOG_DIR = Path("openai_outputs") / timestamp
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    _RESPONSE_COUNT += 1
    data: Dict[str, Any] = {
        "text": getattr(response, "output_text", None),
        "function_calls": [],
    }

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call":
            data["function_calls"].append(
                {
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", None),
                }
            )

    path = _LOG_DIR / f"response_{_RESPONSE_COUNT:03d}.json"
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


@dataclass
class DAG:
    """Simple in-memory representation of a DAG."""

    edges: Dict[str, List[str]] = field(default_factory=dict)
    contents: Dict[str, str] = field(default_factory=dict)

    def add_content(self, node: Node) -> None:
        """Store text for a node if not already present."""
        if node.id not in self.contents:
            self.contents[node.id] = node.text
        self.edges.setdefault(node.id, [])

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

    def add_edge(self, parent: Node, child: Node) -> None:
        self.add_content(parent)
        self.add_content(child)
        if self._has_path(child.id, parent.id):
            raise ValueError(
                f"Adding edge {parent.id!r}->{child.id!r} would create a cycle"
            )
        self.edges[parent.id].append(child.id)

    def to_nested(self, roots: List[str]) -> List[Dict[str, Any]]:
        """Convert the internal edge mapping to a nested structure.

        Each node is represented as an object with ``id``, ``content`` and
        ``children`` fields, where ``children`` is a list of nodes in the same
        format.
        """

        def build(node_id: str, path: Optional[set] = None) -> Dict[str, Any]:
            if path is None:
                path = set()
            if node_id in path:
                raise ValueError(f"Cycle detected at node {node_id!r}")
            children = [
                build(child, path | {node_id}) for child in self.edges.get(node_id, [])
            ]
            return {
                "id": node_id,
                "content": self.contents.get(node_id, ""),
                "children": children,
            }

        return [build(root) for root in roots]


def make_functions(max_fanout: Optional[int]) -> List[Dict[str, Any]]:
    functions = [
        {
            "name": "stop_expansion",
            "description": "Indicate that the given node has no further children.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "ID of the node that should not be expanded further",
                    }
                },
                "required": ["node_id"],
            },
        },
        {
            "name": "new_edges",
            "description": "Return new child nodes for one or more parent nodes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expansions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "node_id": {
                                    "type": "string",
                                    "description": "ID of the parent node being expanded",
                                },
                                "children": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "description": "Text content of the child node",
                                            }
                                        },
                                        "required": ["text"],
                                    },
                                    "description": "Child nodes to attach",
                                },
                            },
                            "required": ["node_id", "children"],
                        },
                        "description": "List of parent nodes and their proposed children",
                    },
                },
                "required": ["expansions"],
            },
        },
    ]
    if max_fanout is not None:
        functions[1]["parameters"]["properties"]["expansions"]["items"]["properties"]["children"]["maxItems"] = max_fanout
    return functions


def make_system_prompt(
    max_fanout: Optional[int], append: Optional[str] = None
) -> str:
    prompt = (
        "You expand nodes in a directed acyclic graph. Each node has an 'id' in the "
        "form 'axiom_node_id-<number>' and some 'text'. When responding, reference "
        "parent nodes by their 'axiom_node_id-<number>'. "
        "Use the 'stop_expansion' function when no further ideas are needed. "
        "Use 'new_edges' to create new child nodes. Either call stop_expansion or "
        "new_edges exactly once. When calling new_edges, supply an array under "
        "'expansions'; each element should contain a 'node_id' and a 'children' "
        "array of objects with 'text'."
    )
    if max_fanout is not None:
        prompt += f" Do not propose more than {max_fanout} child nodes per parent."
    if append:
        prompt += "\n" + append.strip()
    return prompt


async def expand_layer(
    client: "AsyncOpenAI",
    context: str,
    nodes: List[Node],
    model: str,
    max_fanout: Optional[int] = None,
    system_prompt_append: Optional[str] = None,
) -> Dict[str, List[Node]]:
    """Expand all nodes in ``nodes`` using a single batched request.

    Args:
        client: OpenAI client used to make the request.
        context: Conversation context so far.
        nodes: Nodes to expand.
        model: Name of the OpenAI model to use.
        max_fanout: Maximum number of children the model may return per node.
        system_prompt_append: Additional text appended to the system prompt.
    """

    messages = [
        {
            "role": "system",
            "content": make_system_prompt(max_fanout, system_prompt_append),
        }
    ]
    if context:
        messages.append({"role": "user", "content": context})
    messages.append(
        {
            "role": "user",
            "content": "Expand the following nodes:\n"
            + "\n".join(f"{n.id}: {n.text}" for n in nodes),
        }
    )

    functions = make_functions(max_fanout)

    response = await client.responses.create(
        model=model,
        input=messages,
        tools=[{**f, "type": "function"} for f in functions],
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    _store_response(response)

    expansions: Dict[str, List[Node]] = {}
    for item in response.output or []:
        if getattr(item, "type", None) == "function_call":
            name = getattr(item, "name", None)
            arguments = getattr(item, "arguments", "{}")
            payload = json.loads(arguments)
            if name == "new_edges":
                exp_payload = payload.get("expansions")
                if exp_payload is None and payload.get("node_id") is not None:
                    # Backward compatibility for single expansion objects
                    exp_payload = [payload]
                if not exp_payload:
                    continue
                for exp in exp_payload:
                    node_id = exp.get("node_id")
                    if not node_id:
                        continue
                    children_nodes: List[Node] = []
                    for c in exp.get("children", []):
                        text = c.get("text")
                        if text:
                            children_nodes.append(Node(id=_next_id(), text=text))
                    expansions[node_id] = children_nodes
            elif name == "stop_expansion":
                node_id = payload.get("node_id")
                if node_id:
                    expansions[node_id] = []

    return expansions


async def build_dag(
    seeds: List[Node],
    max_nodes: int = 50,
    max_depth: Optional[int] = None,
    max_fanout: Optional[int] = None,
    initial_fanout: Optional[int] = None,
    model: str = "gpt-4o-mini",
    system_prompt_append: Optional[str] = None,
) -> DAG:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required to run this script")

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dag = DAG()
    for seed in seeds:
        dag.add_content(seed)

    queue: List[Tuple[Node, int]] = []
    if max_depth is None or max_depth > 0:
        queue = [(seed, 0) for seed in seeds]

    seen = {seed.id for seed in seeds}
    context_lines: List[str] = []

    while queue and len(dag.edges) < max_nodes:
        layer = queue[0][1]
        nodes = [node for node, _ in queue]
        print(
            f"Layer {layer}: expanding {len(nodes)} node(s)",
            file=sys.stderr,
        )

        current_fanout = (
            initial_fanout if layer == 0 and initial_fanout is not None else max_fanout
        )
        expansions = await expand_layer(
            client,
            "\n".join(context_lines),
            nodes,
            model,
            current_fanout,
            system_prompt_append,
        )

        new_queue: List[Tuple[Node, int]] = []
        context_lines.append(f"Layer {layer}:")
        for parent in nodes:
            children = expansions.get(parent.id, [])
            if current_fanout is not None:
                children = children[:current_fanout]
            context_lines.append(
                f"{parent.id}: {parent.text} -> "
                + (
                    ", ".join(f"{c.id}: {c.text}" for c in children)
                    if children
                    else "(none)"
                )
            )
            for child in children:
                dag.add_edge(parent, child)
                if child.id not in seen:
                    seen.add(child.id)
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

    parser = argparse.ArgumentParser(
        description="Generate a DAG using OpenAI function calls"
    )
    parser.add_argument(
        "seed",
        nargs="*",
        help="Seed node text(s) to start expansion",
    )
    parser.add_argument(
        "--seed-file",
        type=argparse.FileType("r"),
        help="Path to a file whose entire contents form a single seed node",
    )
    parser.add_argument(
        "--seed-stdin",
        action="store_true",
        help="Read a seed from standard input",
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
    parser.add_argument(
        "--initial-fanout",
        type=int,
        default=None,
        help="Maximum number of children for the seed layer",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--sys-prompt-file",
        type=argparse.FileType("r"),
        help="File whose contents are appended to the system prompt",
    )
    args = parser.parse_args()

    def parse_seed(s: str) -> Node:
        return Node(_next_id(), s.strip())

    seeds = [parse_seed(s) for s in args.seed]
    if args.seed_file:
        with args.seed_file as f:
            content = f.read().strip()
            if content:
                seeds.append(parse_seed(content))
    if args.seed_stdin:
        content = sys.stdin.read().strip()
        if content:
            seeds.append(parse_seed(content))

    if not seeds:
        parser.error(
            "No seeds provided. Specify positional seeds or use --seed-file or --seed-stdin."
        )

    sys_prompt_append = None
    if args.sys_prompt_file:
        with args.sys_prompt_file as f:
            sys_prompt_append = f.read()

    dag = asyncio.run(
        build_dag(
            seeds,
            args.max_nodes,
            args.max_depth,
            args.max_fanout,
            args.initial_fanout,
            args.model,
            sys_prompt_append,
        )
    )
    nested = dag.to_nested([s.id for s in seeds])
    print(json.dumps(nested, indent=2))


if __name__ == "__main__":
    main()
