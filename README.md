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

## The process under test

The hidden process is a K-mood chain sitting in front of a damped pendulum. A
mood emits a letter (its own, 70% of the time), the letter becomes a velocity
impulse, and the pendulum's angle is rounded to whole degrees and clamped to
+-90, giving 181 tokens. **The transformer sees nothing but those angle bins.**

The belief is `P(next mood | the letters so far)` from the chain's own forward
algorithm. It is computed before the pendulum and does not depend on it. Two
consequences are worth holding onto:

- the belief's reachable set is a fixed shape -- a triangle for K=3, a
  tetrahedron for K=4 -- which gives a ground-truth picture to check against;
- the belief forgets its own prior after about six letters, so it is close to a
  function of recent input. `experiments/03_myopic.py` exists to measure how
  close, because without that number "the model encodes the belief" cannot be
  told apart from "the model remembers what it just saw".

`mess3` is the canonical three-mood case and is kept as the reference, since a
Mess-4 number is only readable against a Mess-3 one from the same pipeline.
`mess4` differs only in having four moods.

```bash
uv run python scripts/messk_geometry.py    # belief geometry, no training needed
uv run python experiments/01_train.py      # GPU
uv run python experiments/02_probe.py      # belief / mood / velocity by depth
uv run python experiments/03_myopic.py     # the token-window control
uv run python experiments/04_emergence.py  # what is learned, and when
```

GPU work (`01_train.py`) runs on Kaggle T4. Everything else is CPU and runs on
the staging host.

Env knobs: `OUTPUT_DIR`, `CONFIGS`, `TOTAL_TOKENS`, `N_EVAL`, `SEED`, `TAG`,
`WINDOWS`, `FRACTIONS`.
