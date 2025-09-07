import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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


async def expand_node(client: "AsyncOpenAI", node: str) -> Tuple[str, List[str]]:
    """Ask the model to expand a single node."""
    completion = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": node},
        ],
        functions=FUNCTIONS,
        function_call="auto",
    )
    message = completion.choices[0].message
    call = message.get("function_call") or {}
    name = call.get("name")
    arguments = call.get("arguments", "{}")
    payload = json.loads(arguments)

    if name == "new_edges":
        return node, payload.get("children", [])
    return node, []


async def build_dag(seeds: List[str], max_nodes: int = 50) -> DAG:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required to run this script")
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dag = DAG()
    queue = list(seeds)
    seen = set(queue)

    while queue and len(dag.edges) < max_nodes:
        tasks = [expand_node(client, node) for node in queue]
        results = await asyncio.gather(*tasks)
        queue = []
        for parent, children in results:
            for child in children:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
                dag.add_edge(parent, child)
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
