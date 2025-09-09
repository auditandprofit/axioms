# axioms

Utility script for generating a small directed acyclic graph (DAG) using
OpenAI's function calling API.

## Usage

```bash
python dag_generator.py "root node"
```

The script now also accepts a seed prompt from a file using the `--seed-file`
flag. The entire contents of the file are treated as a single seed node:

```bash
python dag_generator.py --seed-file seeds.txt
```

You can further control the graph generation using additional flags:

```bash
python dag_generator.py "root node" --max-depth 3 --max-fanout 2
```

* `--max-depth` limits how deep the expansion proceeds (0 disables expansion).
* `--max-fanout` restricts the number of children added per node.
* `--model` selects the OpenAI model to use (default `gpt-4o-mini`).

During execution the script prints a live status to stderr showing the
current layer being expanded and the number of nodes remaining in that
layer.
