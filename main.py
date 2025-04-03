
import os
import time
import copy
import jax
import wandb
import hydra
from omegaconf import OmegaConf
from jaxtari.jax_seaquest import JaxSeaquest, Renderer_AtraJaxis as SeaquestRenderer
from jaxtari.jax_kangaroo import Kangaroo as JaxKangaroo, Renderer as KangarooRenderer
from symbolic_options.hierarchical_pqn_jaxtari import make_train as make_train_hier
from symbolic_options.pqn_jaxtari import make_train
from jaxtari.wrappers import FlattenObservationWrapper, MultiRewardLogWrapper, AtariWrapper 
from symbolic_options.reward_functions.seaquest import collect_divers_reward, fight_enemies_reward, upward_reward, shaped_reward, learned_meta_policy, llm_meta_policy, conditional_meta_policy, combined_meta_policy, combined_meta_policy_explicit

def outer_make_train(config):

    if config.get("ENV_NAME", None) == "Seaquest":
        # NOTE: the order of the rewards needs to align with the LLM-based meta-policy
        # NOTE: if conditional or combined provide idle_reward (not necessary for llm and learned)
        # this makes sure that there is always a fallback if no rule evaluates to true
        reward_funcs = [fight_enemies_reward, collect_divers_reward, upward_reward]
        # Shaped reward is reward function for meta-policy (not necessary, if meta-policy does not learn) 
        if config.get("META_SHAPED_REWARD", False):
            reward_funcs.append(shaped_reward)
        env = JaxSeaquest(reward_funcs=reward_funcs)
        renderer = SeaquestRenderer()
    elif config.get("ENV_NAME", None) == "Kangaroo":
        env = JaxKangaroo()
        renderer = KangarooRenderer() 
    else:
        raise NotImplementedError(f"Env {config['ENV_NAME']} not implemented.")

    env = FlattenObservationWrapper(env)
    # env = AtariWrapper(env)
    env = MultiRewardLogWrapper(env)

    meta_policy_string = config.get("META_POLICY", "llm")
    if meta_policy_string == "llm":
        meta_policy = llm_meta_policy
    elif meta_policy_string == "learned":
        meta_policy = learned_meta_policy
    elif meta_policy_string == "conditional":
        meta_policy = conditional_meta_policy
    elif meta_policy_string == "combined":
        if config.get("LLM_PRETRAIN", 0) > 0:
            meta_policy = combined_meta_policy_explicit
        else:
            meta_policy = combined_meta_policy
    else:
        raise ValueError("Invalid meta policy")
    
    if config.get("HIERARCHICAL", False):
        make_train_fn = make_train_hier
    else:
        make_train_fn = make_train

    return make_train( config, env, meta_policy, renderer)

def single_run(config):#
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(outer_make_train(config)))
    # time compilation  
    print("Compiling...")
    start = time.time()
    train_vjit.lower(rngs).compile()
    print(f"Compilation took {time.time()-start} seconds.")
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_states = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_map(lambda x: x[i], model_states.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}
    print(default_config)
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "min": 0.00001,
                "max": 0.001,
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="./src/symbolic_options/config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main(None)
