import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvState as CraftaxState

@jax.jit
def nav_to_ladder_reward(prev_state: CraftaxState, state: CraftaxState):
    # navigate to next down_ladder 
    # down == next level (but need to be opened to work)
    next_down_ladder_pos = state.down_ladders[state.player_level]
    prev_down_ladder_pos = prev_state.down_ladders[prev_state.player_level]

    distance = jnp.linalg.norm(
        state.player_position - next_down_ladder_pos
    )
    prev_distance = jnp.linalg.norm(
        prev_state.player_position - prev_down_ladder_pos
    )

    reward = jnp.where(
        distance < prev_distance,
        1.0,
        -1.0,
    ) 
    return reward


@jax.jit
def mine_reward(prev_state: CraftaxState, state: CraftaxState):
    wood_reward = state.inventory.wood - prev_state.inventory.wood
    stone_reward = state.inventory.stone - prev_state.inventory.stone
    coal_reward = state.inventory.coal - prev_state.inventory.coal
    iron_reward = state.inventory.iron - prev_state.inventory.iron
    diamond_reward = state.inventory.diamond - prev_state.inventory.diamond
    return wood_reward + stone_reward + coal_reward + iron_reward + diamond_reward

@jax.jit
def enemies_reward(prev_state: CraftaxState, state: CraftaxState):
    reward = state.monsters_killed[state.player_level] - prev_state.monsters_killed[state.player_level]
    return reward

@jax.jit
def intrinsics_reward(prev_state: CraftaxState, state: CraftaxState):
    health_reward = state.player_health - prev_state.player_health
    food_reward = state.player_food - prev_state.player_food
    drink_reward = state.player_drink - prev_state.player_drink
    energy_reward = state.player_energy - prev_state.player_energy
    mana_reward = state.player_mana - prev_state.player_mana
    return health_reward + food_reward + drink_reward + energy_reward + mana_reward 




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