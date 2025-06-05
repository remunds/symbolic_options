# Symbolic Option Learning

## Installation
### UV project manager
You can either use [uv](https://docs.astral.sh/uv/getting-started/installation/):
- CUDA users probably want to enable GPU acceleration:
```bash
uv add "jax[cuda12]"
``` 
- Now simply run example (e.g. hierarchical seaquest agent with fixed meta-policy):
```bash
uv run main.py +alg=pqn_jaxtari_sea3_hier_llm
```
### Python venv
Instead, you can use venv:
```bash
python3 -m venv .venv
source .venv/bin/activate

python3 -m pip install -U pip
pip3 install -e .
```
- Optionally enable CUDA acceleration:
```bash
pip3 install -U "jax[cuda12]"
```
- Run example:
```bash
python3 main.py +alg=pqn_jaxtari_sea3_hier_llm
```


# Using custom rewards
As an example see how the rewards are included in main.py.
They are implemented in src/reward_functions/.

# Config
See example config under config/alg/pqn_jaxtari_k1.yaml.

For training hierarchical options with different reward functions (designed by LLMs) set META_POLICY to "llm".

# Acknowledgement
PQN implementation from https://github.com/mttga/purejaxql/tree/main