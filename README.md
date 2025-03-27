# Symbolic Option Learning
Note: Currently only Seaquest is supported (next: Kangaroo).

- Install uv: https://docs.astral.sh/uv/getting-started/installation/
- Clone JAXAtari:
```git clone https://github.com/k4ntz/JAXAtari -b lib```
- Setup env:
```uv sync```
- Run example:
```uv run main.py +alg=pqn_jaxtari```

# Using custom rewards
As an example see how the rewards are included in main.py.
They are implemented in src/reward_functions/seaquest.py.

# Config
See example config under config/alg/pqn_jaxtari.yaml.

For training hierarchical options with different reward functions (designed by LLMs) set META_POLICY to "llm".

# Acknowledgement
PQN implementation taken from https://github.com/mttga/purejaxql/tree/main