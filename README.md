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

The complete concise specification is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).
The completed findings are in [`docs/RESULTS.md`](docs/RESULTS.md).

The hidden process is the same four-mood Markov chain in front of each of four damped physical systems: a pendulum, a predator-prey oscillator, a spherical pendulum, and a double pendulum.
A mood remains unchanged with probability 0.7 and otherwise moves to each of the other moods with probability 0.1.
Mood `i` emits letter `i` with probability 0.7 and each other letter with probability 0.1.
The physical system then evolves for ten integration steps and its scalar observable is quantised into 181 tokens.
**The transformer sees only those observation tokens.**

Every system assigns the four letters to distinct, balanced, non-zero physical actions:

| System | Letter 0 | Letter 1 | Letter 2 | Letter 3 |
|---|---|---|---|---|
| Pendulum | strong negative omega kick | weak negative omega kick | weak positive omega kick | strong positive omega kick |
| Predator-prey | prey down | prey up | predator down | predator up |
| Spherical pendulum | north | south | west | east |
| Double pendulum | joint 1 negative | joint 1 positive | joint 2 negative | joint 2 positive |

The pendulum actions are the variance-matched ladder `{-1.643, -0.548, +0.548, +1.643}`.
The other three action sets are opposite directions along their two physical axes.
No HMM letter is a no-op, and every four-action set sums to zero.

The exact analysis target is the predictive belief `P(next mood | letters so far)` from the chain's forward algorithm.
It is computed before the physical system and therefore has the same tetrahedral ground-truth geometry for all four systems.
The comparison asks how much of that common hidden geometry survives each physical channel and becomes linearly readable from the transformer's residual stream.

The Mess-4 belief forgets its prior after approximately seven letters.
`experiments/03_myopic.py` therefore compares the residual-stream probe against a raw recent-token window, because without that control "the model encodes the belief" cannot be separated from "the model remembers what it just saw".

There are four trained models, one per physical system, all at seed 0.
No reseeded runs are part of this experiment.
The model has four transformer blocks, and every analysis reports the embedding plus all four residual-stream depths.
Training checkpoints are saved at 0%, 0.4%, 1%, 2.4%, 6%, 24%, 50%, and 100% of the 500M-token run.

```bash
uv run python scripts/messk_geometry.py    # exact Mess-4 tetrahedron
uv run python experiments/01_train.py      # four GPU training runs
uv run python experiments/02_probe.py      # belief / mood / physical metrics by depth
uv run python experiments/03_myopic.py     # raw-token-window control
uv run python experiments/04_emergence.py  # target R2 over training time
uv run python experiments/05_geometry.py   # tetrahedral geometry by checkpoint and layer
PROJECTION=square uv run python experiments/05_geometry.py  # planar square view
```

GPU work in `01_train.py` runs on Kaggle T4.
All probing and plotting runs on the staging CPU host after the checkpoints land.

Environment knobs are `OUTPUT_DIR`, `CONFIGS`, `TOTAL_TOKENS`, `N_EVAL`, `SEED`, `TAG`, `REPORT_NAME`, `WINDOWS`, and `FRACTIONS`.
Each GPU job writes a distinct report name, and the reports are merged before CPU analysis.
