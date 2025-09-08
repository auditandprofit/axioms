# axioms

Utility script for generating a small directed acyclic graph (DAG) using
OpenAI's function calling API.

## Usage

```bash
python dag_generator.py "root node"
```

The script now also accepts seed prompts from a file using the `--seed-file`
flag. Each line in the file is treated as a separate seed node:

```bash
python dag_generator.py --seed-file seeds.txt
```

You can further control the graph generation using additional flags:

```bash
python dag_generator.py "root node" --max-depth 3 --max-fanout 2
```

* `--max-depth` limits how deep the expansion proceeds (0 disables expansion).
* `--max-fanout` restricts the number of children added per node.
