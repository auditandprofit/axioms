# axioms

Utility script for generating a small directed acyclic graph (DAG) using
OpenAI's function calling API.

## Usage

```bash
python dag_generator.py "root node"
```

Each positional argument is treated as the text for a seed node. The script
assigns sequential identifiers prefixed with `axiom_node_id-` starting at `0` so
you no longer need to specify an ID. You can also provide a seed prompt from a
file using the `--seed-file` flag. The entire contents of the file are treated
as a single seed node:

```bash
python dag_generator.py --seed-file seeds.txt
```

Alternatively, pass the seed via standard input using the `--seed-stdin` flag:

```bash
echo "root node" | python dag_generator.py --seed-stdin
```

You can further control the graph generation using additional flags:

```bash
python dag_generator.py "root node" --max-depth 3 --max-fanout 2
```

* `--max-depth` limits how deep the expansion proceeds (0 disables expansion).
* `--max-fanout` restricts the number of children added per node.
* `--initial-fanout` caps the number of children added for the seed layer only.
* `--model` selects the OpenAI model to use (default `gpt-4o-mini`).
* `--sys-prompt-file` appends the contents of a file to the system prompt.
* `--reasoning-effort` forwards a reasoning effort value to the OpenAI API.
* `--service-tier` sets the service tier used for API requests.

To append custom instructions to the system prompt:

```bash
python dag_generator.py "root node" --sys-prompt-file extra_prompt.txt
```

During execution the script prints a live status to stderr showing the
current layer being expanded and the number of nodes remaining in that
layer. The final tree is written to standard output *and* persisted to
`openai_outputs/<timestamp>/final_tree.json` alongside the raw response
logs for later inspection.
