# Goals:
# - load agent
# - collect (S, A, Q_distribution) from jaxatari using agent's policy
# - compute (S, A, weight)
# - save
import hydra
from omegaconf import OmegaConf
from symbolic_options.purejaxql.save_load import load_params
import os
import jax
import jax.numpy as jnp

from jaxatari.games.jax_pong import JaxPong
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper, MultiRewardLogWrapper
from symbolic_options.reward_functions.pong import track_and_align, return_shot, defensive_positioning, llm_meta_policy, combined_meta_policy, learned_meta_policy, conditional_meta_policy 

def create_env(config):
    if config.get("ALG_NAME", "") == "pqn_hier_llm":
        reward_funcs = [track_and_align, return_shot, defensive_positioning]
    else:
        reward_funcs = []
    env = JaxPong(reward_funcs=reward_funcs)
    env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
    env = ObjectCentricWrapper(env)
    env = FlattenObservationWrapper(env)
    return env

def load_network_params(config):
    if config.get("LOAD_PATH", None) is not None:
        alg_name = config["ALG_NAME"]
        env_name = config["ENV_NAME"]
        load_dir = os.path.join(config["LOAD_PATH"], env_name)
        print(f"Loading parameters from {load_dir} for algorithm {alg_name} on environment {env_name}")
        params = []
        batch_stats = []
        for i in range(config["NUM_SEEDS"]):
            load_path = os.path.join(
                load_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            inner_params = None
            inner_bs = None
            # check if file exists
            if os.path.exists(load_path):
                print(f"Loading params from {load_path}")
                inner_params = load_params(load_path)
                inner_bs = load_params(load_path.replace(".safetensors", "_bs.safetensors"))
            else:
                print(f"File {load_path} does not exist, using random params")
            params.append(inner_params)
            batch_stats.append(inner_bs)

        # transform to pytree
        params = jax.tree_util.tree_map(
            lambda *xs: jnp.squeeze(jnp.stack(xs)), *params
        )
        batch_stats = jax.tree_util.tree_map(
            lambda *xs: jnp.squeeze(jnp.stack(xs)), *batch_stats
        )

        return params, batch_stats

    return None, None

def run_agent(config):
    config = {**config, **config["alg"]}
    if config.get("ALG_NAME", "") == "pqn_hier_llm":
        from symbolic_options.hierarchical_pqn_jaxtari import QNetwork, CustomTrainState
        #TODO: this is just pong currently
        from symbolic_options.reward_functions.pong import llm_meta_policy
    else: # for now default to pqn
        from symbolic_options.pqn_jaxtari import QNetwork, CustomTrainState
        # always use option0 (default)
        llm_meta_policy = lambda a,b,c,d: jnp.array([0]) 

    import numpy as np
    import optax


    params, batch_stats = load_network_params(config)
    if params is None or batch_stats is None:
        raise ValueError("No parameters loaded. Check your configuration and paths.")
    
    rng = jax.random.PRNGKey(config["SEED"])

    env = create_env(config)

    config["NUM_AGENTS"] = len(env.reward_funcs)
    config["OBS_SHAPE"] = env.observation_space().shape
    config["NUM_ACTIONS"] = env.action_space().n

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
        jax.random.split(rng, n_envs)
    )
    vmap_step = lambda env_state, action: jax.vmap(
        env.step
    )(env_state, action)

    network = QNetwork(
        action_dim=config["NUM_ACTIONS"],
        hidden_size=config.get("HIDDEN_SIZE", 64),
        num_layers=config.get("NUM_LAYERS", 3),
        norm_type=config["NORM_TYPE"],
        norm_input=config.get("NORM_INPUT", False),
    )

    def create_agent(rng, params, batch_stats, network, lr):
        obs_len = np.prod(config["OBS_SHAPE"])
        init_x = jnp.zeros(obs_len)
        network_variables = network.init(rng, init_x, train=False)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_variables["params"] if params is None else params,
            batch_stats=network_variables["batch_stats"] if batch_stats is None else batch_stats,
            tx=tx,
        )
        return train_state

    num_agents = config.get("NUM_AGENTS", 1)
    print(f"Number of agents (subpolicies): {num_agents}")

    if num_agents == 0 or num_agents == 1:
        train_states: CustomTrainState = create_agent(rng, params, batch_stats, network, 0.0)
        # add a vmap dimension for consistency
        train_states = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), train_states)
    else:
        rng_keys = jax.random.split(rng, num_agents)
        train_states: CustomTrainState = jax.vmap(create_agent, in_axes=(0, 0, 0, None, None))(rng_keys, params, batch_stats, network, 0.0)

    def get_action_and_qvals(train_states, obs, env_state):
        def compute_actions_subpolicy(obs, train_state):
            q_vals = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                obs,
                train=False,
            )
            greedy_action = jnp.argmax(q_vals, axis=-1)
            return greedy_action, q_vals
        # compute actions and qvals for all subpolicies
        all_actions, all_q_vals = jax.vmap(compute_actions_subpolicy, in_axes=(None, 0))(obs, train_states)

        # use meta-policy to get the currently active subpolicy
        meta_q_vals = llm_meta_policy(None, None, obs, env_state) 
        active_subpolicy = jnp.argmax(meta_q_vals, axis=-1)
        # get corresponding action and q-vals
        action = all_actions[active_subpolicy, jnp.arange(config["NUM_ENVS"])]
        qvals = all_q_vals[active_subpolicy, jnp.arange(config["NUM_ENVS"])]
        return action, qvals


    def scan_step_fn(carry, x):
        current_obs, current_env_state = carry

        # Get action and q-values for the current observation and environment state
        action, qvals = get_action_and_qvals(train_states, current_obs, current_env_state)

        # Step the environment with the produced action
        new_obs, new_env_state, reward, new_done, info = vmap_step(current_env_state, action)

        # The carry for the next iteration is the new observation and environment state
        next_carry = (new_obs, new_env_state)

        # We want to collect action and qvals from each step
        outputs_to_collect = (current_obs, new_done, action, qvals, reward)

        return next_carry, outputs_to_collect

    # Perform the scan over the environment steps
    # We use jnp.arange(config["NUM_STEPS"]) as a dummy sequence for 'xs'
    # because the state evolution depends on the previous step's output.
    init_obs, init_env_state = vmap_reset(config["NUM_ENVS"])(rng)
    final_carry, (obs, dones, actions, qvals, rewards) = jax.lax.scan(
        scan_step_fn,
        init=(init_obs, init_env_state),
        xs=jnp.arange(config["NUM_STEPS"])
    )
    return obs, dones, actions, qvals, rewards

@hydra.main(version_base=None, config_path="./src/symbolic_options/config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    all_obs, all_dones, all_actions, all_qvals, rewards = run_agent(config)
    #obs: (n_steps, n_envs, obs_shape)
    #dones: (n_steps, n_envs)
    #actions: (n_steps, n_envs)
    #qvals: (n_steps, n_envs, n_actions)

    print("Observations collected:", all_obs.shape)
    print("Dones collected:", all_dones.shape)
    print("Actions collected:", all_actions.shape)
    print("Q-values collected:", all_qvals.shape)
    print("Rewards collected:", rewards.shape)
    print("Average env return: ", jnp.mean(jnp.sum(rewards, axis=0)))
    print("average dones per env: ", jnp.mean(jnp.sum(all_dones, axis=0)))

    # create tuple (S, A, Q_distribution)
    S = all_obs.reshape(-1, all_obs.shape[-1])  # (n_steps * n_envs, obs_shape)
    A = all_actions.reshape(-1)  # (n_steps * n_envs,)
    Q_distribution = all_qvals.reshape(-1, all_qvals.shape[-1])  # (n_steps * n_envs, n_actions)
    print("S shape:", S.shape)
    print("A shape:", A.shape)
    print("Q_distribution shape:", Q_distribution.shape)


if __name__ == "__main__":
    main(None)