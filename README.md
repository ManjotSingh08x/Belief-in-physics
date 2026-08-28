## Installation and Setup
We will be using uv as our package manager. To sync and setup dependancies, ensure `uv` is installed
To build and work do :- 

```bash
uv sync
```
To run scripts and notebooks, use 

```bash
uv run python train.py
uv run jupyter lab
```

To add or remove dependancies, use format 

```bash
uv add torch 
uv remove torch
```

- Also, always commit pyproject.toml and uv.lock together in the same PR.
- Work in seperate branches and do not commit directly to `main`.

## Work setup
all common code should be written in `models/` directory or in `physics` directory. it should be in the form of packages which can be easily imported.

Do not define models, datasets, dataloaders, etc. inside the experiment script. Rather define them in `models/` directory or `physics/` directory and import them.

## Experiments
All experiments should begin with a number at the start. the outputs, temporary values, results, model weights should be stored in `experiments/` directory under a folder named `experiments/outputs-{number}`. This directory would be gitignored by default. If we need persistent storage, we should move it to `experiments/results-{number}` subdirectory and commit it.

In each experiment file, we should define constants and variables at the beginning of the file. This includes `EMBED_DIM`, `NUM_LAYERS`, `NUM_HEADS`, `THETA_RES`, `MAX_SEQ_LEN`, `OUTPUT_DIR`, etc. Use tqdm wherever possible for easy working. keep code modular and use standard OOPs principles. 

