## Hyperparameters

`docs/HYPERPARAM.md` is the single inventory of every adjustable parameter in this
experiment: the chain, the observation channel, all four physical systems, the model,
the training recipe, every analysis constant, and the infrastructure settings.

Rules:
- **Change a value, update `docs/HYPERPARAM.md` in the same commit.** This includes
  changing a default, adding a new constant or env knob, renaming one, or removing one.
  A commit that touches a hyperparameter without touching that file is incomplete.
- Record the value that is actually in use. Where code default and current run differ,
  the table carries both columns rather than silently overwriting the default.
- When a value was chosen by measurement, record the measurement next to it, not just
  the number. The sphere `delta_v` and the pendulum `gamma` entries are the pattern:
  they say what clipping rate the chosen value produces.
- Mark derived quantities as derived and give the formula, so a change to an input
  does not silently break an assumption elsewhere.
- Before adding a new tunable, check whether it belongs to an existing group in that
  file rather than creating a new section.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
