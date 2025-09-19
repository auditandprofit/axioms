import asyncio
import copy
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import (
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        AsyncOpenAI,
        OpenAIError,
        RateLimitError,
    )
except ImportError:
    AsyncOpenAI = None

    class _OpenAIErrorFallback(Exception):
        """Fallback error base when the OpenAI package is unavailable."""

    class _APIErrorFallback(_OpenAIErrorFallback):
        pass

    class _APIStatusErrorFallback(_APIErrorFallback):
        status_code: Optional[int] = None

    class _APIConnectionErrorFallback(_APIErrorFallback):
        pass

    class _APITimeoutErrorFallback(_APIConnectionErrorFallback):
        pass

    class _RateLimitErrorFallback(_APIStatusErrorFallback):
        pass

    OpenAIError = _OpenAIErrorFallback
    APIError = _APIErrorFallback
    APIStatusError = _APIStatusErrorFallback
    APIConnectionError = _APIConnectionErrorFallback
    APITimeoutError = _APITimeoutErrorFallback
    RateLimitError = _RateLimitErrorFallback


_LOG_DIR: Optional[Path] = None
_RESPONSE_COUNT = 0

# Counter used to assign sequential IDs to nodes
_NODE_COUNTER = 0


class OpenAIMaxRetriesExceededError(RuntimeError):
    """Raised when OpenAI requests exhaust the configured retry budget."""

    def __init__(
        self,
        *,
        attempts: int,
        max_retries: int,
        last_exception: BaseException,
    ) -> None:
        message = (
            "OpenAI request failed after "
            f"{attempts} attempt(s); max retries is {max_retries}. "
            f"Last error: {last_exception}"
        )
        super().__init__(message)
        self.attempts = attempts
        self.max_retries = max_retries
        self.last_exception = last_exception


def _get_env_float(name: str, default: float) -> float:
    """Return a float from ``os.environ`` falling back to ``default``."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_env_int(name: str, default: int) -> int:
    """Return an integer from ``os.environ`` falling back to ``default``."""

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_OPENAI_REQUEST_TIMEOUT = max(0.0, _get_env_float("OPENAI_REQUEST_TIMEOUT", 60.0))
_OPENAI_MAX_RETRIES = max(0, _get_env_int("OPENAI_MAX_RETRIES", 5))
_OPENAI_INITIAL_RETRY_DELAY = max(0.0, _get_env_float("OPENAI_INITIAL_RETRY_DELAY", 1.0))
_OPENAI_MAX_RETRY_DELAY = max(
    _OPENAI_INITIAL_RETRY_DELAY, _get_env_float("OPENAI_MAX_RETRY_DELAY", 30.0)
)
_OPENAI_RETRY_JITTER = max(0.0, _get_env_float("OPENAI_RETRY_JITTER", 0.1))


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


@dataclass
class FlowContext:
    """Represents the state required to continue expanding a single path."""

    node: Node
    depth: int
    history: List[Dict[str, Any]] = field(default_factory=list)


def _layer_to_payload(
    layer_index: int,
    nodes: List[Node],
    parent_lookup: Dict[str, str],
) -> Dict[str, Any]:
    """Return a JSON-serialisable structure describing a layer of nodes."""

    payload: Dict[str, Any] = {
        "layer_index": layer_index,
        "nodes": [],
    }

    for node in nodes:
        node_entry: Dict[str, Any] = {
            "id": node.id,
            "text": node.text,
        }
        if layer_index > 0:
            parent_id = parent_lookup.get(node.id)
            if parent_id is not None:
                node_entry["parent_id"] = parent_id
        payload["nodes"].append(node_entry)

    return payload


def _ensure_log_dir() -> Path:
    """Return the directory used to persist request/response data."""

    global _LOG_DIR
    if _LOG_DIR is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _LOG_DIR = Path("openai_outputs") / timestamp
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def _store_response(response: Any) -> None:
    """Persist function call and text data from a response to disk."""

    global _RESPONSE_COUNT

    log_dir = _ensure_log_dir()

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

    path = log_dir / f"response_{_RESPONSE_COUNT:03d}.json"
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


def _store_final_tree(tree: Any) -> None:
    """Persist the final nested tree representation to disk."""

    try:
        log_dir = _ensure_log_dir()
    except Exception:
        return

    path = log_dir / "final_tree.json"
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(tree, fh, indent=2)
    except Exception:
        pass


def _store_error_messages(messages: List[str]) -> None:
    """Append the given error messages to a log file in the output directory."""

    if not messages:
        return

    try:
        log_dir = _ensure_log_dir()
    except Exception:
        return

    path = log_dir / "errors.log"
    timestamp = datetime.now().isoformat()

    try:
        with path.open("a", encoding="utf-8") as fh:
            for message in messages:
                fh.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def _is_retryable_openai_error(exc: BaseException) -> bool:
    """Return ``True`` if the given OpenAI exception should be retried."""

    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            return True
        if status_code == 429:
            return True
        if 500 <= status_code < 600:
            return True
        return False
    return False


async def _create_response_with_retry(
    client: "AsyncOpenAI", request_kwargs: Dict[str, Any]
) -> Any:
    """Call ``client.responses.create`` with retries and a request timeout."""

    options = dict(request_kwargs)
    options.setdefault("timeout", _OPENAI_REQUEST_TIMEOUT)

    retries = 0
    while True:
        try:
            return await client.responses.create(**options)
        except OpenAIError as exc:
            if not _is_retryable_openai_error(exc):
                raise
            if retries >= _OPENAI_MAX_RETRIES:
                attempts = retries + 1
                raise OpenAIMaxRetriesExceededError(
                    attempts=attempts,
                    max_retries=_OPENAI_MAX_RETRIES,
                    last_exception=exc,
                ) from exc

            delay = min(
                _OPENAI_MAX_RETRY_DELAY,
                _OPENAI_INITIAL_RETRY_DELAY * (2**retries),
            )
            jitter = random.uniform(0.0, delay * _OPENAI_RETRY_JITTER)
            sleep_for = delay + jitter
            if sleep_for > 0:
                print(
                    (
                        f"OpenAI request failed ({exc!s}); retrying in "
                        f"{sleep_for:.2f}s"
                    ),
                    file=sys.stderr,
                )
                await asyncio.sleep(sleep_for)
            retries += 1
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


def make_functions(forced_fanout: Optional[int]) -> List[Dict[str, Any]]:
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
    if forced_fanout is not None:
        child_schema = (
            functions[1][
                "parameters"
            ]["properties"]["expansions"]["items"]["properties"]["children"]
        )
        child_schema["description"] = (
            f"Child nodes to attach; must contain exactly {forced_fanout} entries"
        )
        child_schema["minItems"] = forced_fanout
        child_schema["maxItems"] = forced_fanout
    return functions


def make_system_prompt(
    forced_fanout: Optional[int], append: Optional[str] = None
) -> str:
    prompt = (
        "You expand nodes in a directed acyclic graph. Each node has an 'id' in the "
        "form 'axiom_node_id-<number>' and some 'text'. When responding, reference "
        "parent nodes by their 'axiom_node_id-<number>'. Process every node listed "
        "in the most recent user message describing the current layer. For each "
        "node: if it needs more ideas, call 'new_edges' and include that node in the "
        "'expansions' array with a 'children' array of objects containing only a "
        "'text' field for each child (IDs are assigned automatically). If the node "
        "should not receive more children, call 'stop_expansion' with that node's "
        "ID. You may call the tools multiple times, but by the end of your response "
        "every node from the current layer must have been handled exactly once—"
        "either via 'new_edges' or 'stop_expansion'—and you must not invoke both "
        "tools for the same node. When creating children, obey any stated limits on "
        "the number of children per parent."
    )
    if forced_fanout is not None:
        prompt += (
            f" When expanding a node, you must propose exactly {forced_fanout} child "
            "nodes (no more, no fewer)."
        )
    if append:
        prompt += "\n" + append.strip()
    return prompt


async def expand_layer(
    client: "AsyncOpenAI",
    history_layers: List[Dict[str, Any]],
    current_layer_payload: Dict[str, Any],
    model: str,
    forced_fanout: Optional[int] = None,
    system_prompt_append: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    service_tier: Optional[str] = None,
) -> Dict[str, List[Node]]:
    """Expand the nodes described by ``current_layer_payload`` in one request.

    Args:
        client: OpenAI client used to make the request.
        history_layers: Previously generated layers to replay to the model.
        current_layer_payload: Description of the layer currently being expanded.
        model: Name of the OpenAI model to use.
        forced_fanout: Exact number of children the model must return per node.
        system_prompt_append: Additional text appended to the system prompt.
    """

    messages = [
        {
            "role": "system",
            "content": make_system_prompt(forced_fanout, system_prompt_append),
        }
    ]
    for layer_payload in history_layers:
        messages.append(
            {
                "role": "user",
                "content": json.dumps(layer_payload, indent=2, ensure_ascii=False),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                current_layer_payload, indent=2, ensure_ascii=False
            ),
        }
    )

    functions = make_functions(forced_fanout)

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "tools": [{**f, "type": "function"} for f in functions],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    if reasoning_effort:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    if service_tier:
        request_kwargs["service_tier"] = service_tier

    response = await _create_response_with_retry(client, request_kwargs)

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
                    node_id_raw = exp.get("node_id")
                    if not node_id_raw:
                        continue
                    node_id = str(node_id_raw).strip()
                    if not node_id:
                        continue
                    children_nodes: List[Node] = []
                    for c in exp.get("children", []):
                        text = c.get("text")
                        if text:
                            children_nodes.append(Node(id=_next_id(), text=text))
                    if (
                        forced_fanout is not None
                        and children_nodes
                        and len(children_nodes) != forced_fanout
                    ):
                        raise ValueError(
                            "new_edges for node"
                            f" {node_id!r} returned {len(children_nodes)} child nodes"
                            f" (expected {forced_fanout})"
                        )
                    expansions[node_id] = children_nodes
            elif name == "stop_expansion":
                node_id_raw = payload.get("node_id")
                if not node_id_raw:
                    continue
                node_id = str(node_id_raw).strip()
                if node_id:
                    expansions[node_id] = []

    return expansions


async def build_dag(
    seeds: List[Node],
    max_nodes: int = 50,
    max_depth: Optional[int] = None,
    forced_fanout: Optional[int] = None,
    initial_forced_fanout: Optional[int] = None,
    model: str = "gpt-4o-mini",
    system_prompt_append: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    service_tier: Optional[str] = None,
) -> DAG:
    if AsyncOpenAI is None:
        raise RuntimeError("openai package is required to run this script")

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    dag = DAG()
    for seed in seeds:
        dag.add_content(seed)

    seen = {seed.id for seed in seeds}
    node_parents: Dict[str, str] = {}

    initial_flows: List[FlowContext] = []
    if max_depth is None or max_depth > 0:
        for seed in seeds:
            initial_flows.append(FlowContext(node=seed, depth=0))

    async def expand_context(context: FlowContext) -> None:
        if len(dag.edges) >= max_nodes:
            return
        if max_depth is not None and context.depth >= max_depth:
            return

        base_payload = _layer_to_payload(context.depth, [context.node], node_parents)
        request_payload = copy.deepcopy(base_payload)
        request_payload["action"] = "expand"

        print(
            f"Depth {context.depth}: expanding path ending at {context.node.id}",
            file=sys.stderr,
        )

        current_fanout = (
            initial_forced_fanout
            if context.depth == 0 and initial_forced_fanout is not None
            else forced_fanout
        )

        expansions = await expand_layer(
            client,
            context.history,
            request_payload,
            model,
            current_fanout,
            system_prompt_append,
            reasoning_effort,
            service_tier,
        )

        history_for_children = context.history + [copy.deepcopy(base_payload)]

        if current_fanout is not None and context.node.id not in expansions:
            raise ValueError(
                "Expected forced fanout expansion for node"
                f" {context.node.id!r} but received no tool call."
            )

        children = expansions.get(context.node.id, [])

        if (
            current_fanout is not None
            and children
            and len(children) != current_fanout
        ):
            raise ValueError(
                f"Node {context.node.id!r} returned {len(children)} child nodes"
                f" (expected {current_fanout})."
            )

        if not children:
            return

        next_depth = context.depth + 1

        scheduled_children: List[FlowContext] = []
        skipped_due_to_depth = 0
        for child in children:
            if len(dag.edges) >= max_nodes:
                break
            dag.add_edge(context.node, child)
            if child.id not in seen:
                seen.add(child.id)
                node_parents[child.id] = context.node.id
                if max_depth is None or next_depth < max_depth:
                    scheduled_children.append(
                        FlowContext(
                            node=child,
                            depth=next_depth,
                            history=list(history_for_children),
                        )
                    )
                else:
                    skipped_due_to_depth += 1

        if children:
            message = (
                f"Queued {len(scheduled_children)} flow(s) "
                f"at depth {next_depth} from {context.node.id}"
            )
            if skipped_due_to_depth:
                suffix = "child" if skipped_due_to_depth == 1 else "children"
                message += (
                    f" (max depth reached for {skipped_due_to_depth} {suffix})"
                )
            print(message, file=sys.stderr)

        if scheduled_children:
            await asyncio.gather(*(expand_context(child) for child in scheduled_children))

    if initial_flows:
        await asyncio.gather(*(expand_context(flow) for flow in initial_flows))

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
        "--forced-fanout",
        "--max-fanout",
        dest="forced_fanout",
        type=int,
        default=None,
        help=(
            "Exact number of children each expanded node must produce"
            " (alias: --max-fanout)"
        ),
    )
    parser.add_argument(
        "--initial-forced-fanout",
        "--initial-fanout",
        dest="initial_forced_fanout",
        type=int,
        default=None,
        help=(
            "Exact number of children nodes in the seed layer must produce"
            " (alias: --initial-fanout)"
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help=(
            "Request timeout in seconds for OpenAI API calls. Overrides the "
            "OPENAI_REQUEST_TIMEOUT environment variable."
        ),
    )
    parser.add_argument(
        "--sys-prompt-file",
        type=argparse.FileType("r"),
        help="File whose contents are appended to the system prompt",
    )
    parser.add_argument(
        "--reasoning-effort",
        help="Value for the reasoning effort passed to the OpenAI API",
    )
    parser.add_argument(
        "--service-tier",
        help="Service tier value forwarded to the OpenAI API",
    )
    args = parser.parse_args()

    if args.request_timeout is not None:
        value = max(0.0, float(args.request_timeout))
        global _OPENAI_REQUEST_TIMEOUT
        _OPENAI_REQUEST_TIMEOUT = value

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

    try:
        dag = asyncio.run(
            build_dag(
                seeds,
                args.max_nodes,
                args.max_depth,
                args.forced_fanout,
                args.initial_forced_fanout,
                args.model,
                sys_prompt_append,
                args.reasoning_effort,
                args.service_tier,
            )
        )
    except OpenAIMaxRetriesExceededError as exc:
        exit_code = getattr(os, "EX_TEMPFAIL", 75)
        messages = [f"{type(exc).__name__}: {exc}"]
        last_exception = getattr(exc, "last_exception", None)
        if last_exception is not None:
            messages.append(f"{type(last_exception).__name__}: {last_exception}")
        _store_error_messages(messages)
        print(exc, file=sys.stderr)
        sys.exit(exit_code)
    nested = dag.to_nested([s.id for s in seeds])
    _store_final_tree(nested)
    print(json.dumps(nested, indent=2))


if __name__ == "__main__":
    main()
