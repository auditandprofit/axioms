import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


@dataclass
class DAG:
    """Simple in-memory representation of a DAG."""

    edges: Dict[str, List[str]] = field(default_factory=dict)

    def add_edge(self, parent: str, child: str) -> None:
        self.edges.setdefault(parent, []).append(child)
        self.edges.setdefault(child, [])


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
        model="gpt-4o-mini",
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


async def build_dag(seeds: List[str], max_nodes: int = 50) -> DAG:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required to run this script")
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dag = DAG()
    queue: List[Tuple[str, Optional[str]]] = [(seed, None) for seed in seeds]
    seen = set(seeds)

    while queue and len(dag.edges) < max_nodes:
        tasks = [expand_node(client, base, node) for base, node in queue]
        results = await asyncio.gather(*tasks)
        new_queue: List[Tuple[str, Optional[str]]] = []
        for (base, _), (parent, children) in zip(queue, results):
            for child in children:
                if child not in seen:
                    seen.add(child)
                    new_queue.append((base, child))
                dag.add_edge(parent, child)
        queue = new_queue
    return dag


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a DAG using OpenAI function calls")
    parser.add_argument("seed", nargs="+", help="Seed node(s) to start expansion")
    parser.add_argument("--max-nodes", type=int, default=50, help="Maximum number of nodes to generate")
    args = parser.parse_args()

    dag = asyncio.run(build_dag(args.seed, args.max_nodes))
    print(json.dumps(dag.edges, indent=2))


if __name__ == "__main__":
    main()
