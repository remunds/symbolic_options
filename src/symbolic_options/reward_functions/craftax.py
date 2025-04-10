import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvState as CraftaxState

@jax.jit
def mine_reward(prev_state: CraftaxState, state: CraftaxState):
    reward = 0 
    return reward

@jax.jit
def llm_meta_policy(network, meta_train_state, last_obs, env_state: CraftaxState):
    """
    Mutually exclusive.
    """
    # N_envs, n_actions
    zeros = jnp.zeros((1024, 43))
    return zeros

# @jax.jit
def conditional_meta_policy(network, meta_train_state, last_obs, env_state: CraftaxState):
    """
    Same as llm_meta_policy but not mutually exclusive.
    """
    q_vals = jnp.zeros((last_obs.shape[0]))
    return q_vals

# @jax.jit
def learned_meta_policy(network, meta_train_state, last_obs, env_state: CraftaxState):#
    q_vals = network.apply(
        {
            "params": meta_train_state.params,
            "batch_stats": meta_train_state.batch_stats,
        },
        last_obs,
        train=False,
    )
    return q_vals

# @jax.jit
def combined_meta_policy(network, meta_train_state, last_obs, env_state: CraftaxState):
    # combine learned and conditional meta policy
    conditional_q_vals = llm_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return combined_q_vals

def combined_meta_policy_explicit(network, meta_train_state, last_obs, env_state: CraftaxState):
    llm_q_vals = llm_meta_policy(network, meta_train_state, last_obs, env_state)
    conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return llm_q_vals, combined_q_vals