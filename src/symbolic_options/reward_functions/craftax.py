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


# pseudo code meta-policy from deepseek-r1
# def select_skill(state: EnvState) -> int:
#     # Skill IDs: 0=Survival, 1=Combat, 2=Resource, 3=Crafting, 4=Explore, 5=LevelProgression

#     # 1. Survival emergencies first
#     if (state.player_health < 20 or 
#         state.player_food < 10 or 
#         state.player_drink < 10 or 
#         state.player_energy < 10):
#         return 0  # Survival Management

#     # 2. Combat priority: attack mobs in proximity
#     if has_nearby_enemies(state):
#         return 1  # Combat Engagement

#     # 3. Level progression: find ladder after killing 8 mobs (exclude overworld)
#     if (state.player_level > 0 and 
#         state.monsters_killed[state.player_level] >= 8):
#         return 5  # Level Progression

#     # 4. Collect resources if lacking tools/materials
#     if (state.inventory.pickaxe == 0 or 
#         state.inventory.sword == 0 or 
#         state.inventory.wood < 5 or 
#         state.inventory.stone < 5):
#         return 2  # Resource Collection

#     # 5. Craft better gear if possible
#     if can_craft_pickaxe(state.inventory) or can_craft_sword(state.inventory):
#         return 3  # Crafting/Upgrading

#     # 6. Default: explore the map
#     return 4  # Exploration

# # Helper functions (simplified for brevity)
# def has_nearby_enemies(state):
#     # Check if any hostile mob is adjacent to the player
#     player_pos = state.player_position
#     for mob_type in [state.melee_mobs, state.ranged_mobs]:
#         for i in range(mob_type.mask.shape[0]):
#             if mob_type.mask[i] and manhattan_distance(player_pos, mob_type.position[i]) <= 2:
#                 return True
#     return False

# def can_craft_pickaxe(inventory):
#     # Check if stone pickaxe can be crafted (assumes tiered system)
#     return inventory.stone >= 3 and inventory.pickaxe < 2

# def can_craft_sword(inventory):
#     # Check if basic sword can be crafted
#     return inventory.wood >= 2 and inventory.stone >= 1 and inventory.sword == 0