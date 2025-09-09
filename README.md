# axioms

Utility script for generating a small directed acyclic graph (DAG) using
OpenAI's function calling API.

## Usage

```bash
python dag_generator.py "root node"
```

Each positional argument is treated as the text for a seed node. The script
assigns simple numeric identifiers starting at `0` so you no longer need to
specify an ID. You can also provide a seed prompt from a file using the
`--seed-file` flag. The entire contents of the file are treated as a single seed
node:

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
* `--model` selects the OpenAI model to use (default `gpt-4o-mini`).

During execution the script prints a live status to stderr showing the
current layer being expanded and the number of nodes remaining in that
layer.
