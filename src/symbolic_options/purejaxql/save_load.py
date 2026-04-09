import os
import json
import jax
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Dict, Union

def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename:Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


def save(
    params: Dict,
    config: Dict,
    save_dir: Union[str, os.PathLike],
    save_name: str,
    vmaps: int = 1,
) -> None:
    """Compatibility wrapper used by mujoco trainers.

    Saves one safetensors file per vmapped seed and a JSON config snapshot.
    """
    os.makedirs(save_dir, exist_ok=True)

    if vmaps <= 1:
        save_params(params, os.path.join(save_dir, f"{save_name}.safetensors"))
    else:
        for i in range(vmaps):
            seed_params = jax.tree_util.tree_map(lambda x: x[i], params)
            save_params(
                seed_params,
                os.path.join(save_dir, f"{save_name}_seed{i}.safetensors"),
            )

    with open(os.path.join(save_dir, f"{save_name}_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)