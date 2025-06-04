# Symbolic Option Learning
- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Setup env:
```uv sync```
- Run example (e.g. hierarchical seaquest agent with fixed meta-policy):
```uv run main.py +alg=pqn_jaxtari_sea3_hier_llm```

# Using custom rewards
As an example see how the rewards are included in main.py.
They are implemented in src/reward_functions/.

# Config
See example config under config/alg/pqn_jaxtari_k1.yaml.

For training hierarchical options with different reward functions (designed by LLMs) set META_POLICY to "llm".

# Acknowledgement
PQN implementation from https://github.com/mttga/purejaxql/tree/main