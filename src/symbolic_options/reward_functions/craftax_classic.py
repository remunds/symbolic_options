from functools import partial
import jax
import jax.numpy as jnp
from craftax.craftax_classic.envs.craftax_state import EnvState as CraftaxState, Inventory
from symbolic_options.purejaxql.craftax_wrappers import MultiRewardLogEnvState

@jax.jit
def survival_reward(prev_state: CraftaxState, state: CraftaxState):
    # Reward for survival management (player needs)
    health_reward = state.player_health - prev_state.player_health
    food_reward = state.player_food - prev_state.player_food
    drink_reward = state.player_drink - prev_state.player_drink
    energy_reward = state.player_energy - prev_state.player_energy
    return 5*health_reward + 2*food_reward + 2*drink_reward + 0.5*energy_reward

@jax.jit
def combat_reward(prev_state: CraftaxState, state: CraftaxState):
    # NOTE: this might require additional incentive like rewarding hitting/using bow/...
    # Reward for combat engagement (killing mobs)
    # killed_a_mob = jnp.where(prev_state.zombies.mask.sum() > state.zombies.mask.sum(), 1, 0)
    reduced_zombie_health = jnp.where(prev_state.zombies.health.sum() > state.zombies.health.sum(), 5, 0)
    reduced_skeleton_health = jnp.where(prev_state.skeletons.health.sum() > state.skeletons.health.sum(), 5, 0)
    reduced_cow_health = jnp.where(prev_state.cows.health.sum() > state.cows.health.sum(), 5, 0)

    # Punish loss of health
    player_health_reward = state.player_health - prev_state.player_health

    return reduced_zombie_health + reduced_skeleton_health + reduced_cow_health + 0.5*player_health_reward 

@jax.jit
def resource_collection_reward(prev_state: CraftaxState, state: CraftaxState):
    # Reward for resource collection (wood, stone, coal, iron, diamond)
    wood_reward = state.inventory.wood - prev_state.inventory.wood
    wood_reward = jnp.where(state.inventory.wood < 20, wood_reward, 0)
    stone_reward = state.inventory.stone - prev_state.inventory.stone
    stone_reward = jnp.where(state.inventory.stone < 10, stone_reward, 0)
    coal_reward = state.inventory.coal - prev_state.inventory.coal
    coal_reward = jnp.where(state.inventory.coal < 10, coal_reward, 0)
    iron_reward = state.inventory.iron - prev_state.inventory.iron
    iron_reward = jnp.where(state.inventory.iron < 10, iron_reward, 0)
    diamond_reward = state.inventory.diamond - prev_state.inventory.diamond
    resource_reward = wood_reward + 2 * stone_reward + 2 * coal_reward + 3 * iron_reward + 5 * diamond_reward
    return resource_reward

def should_craft_pickaxe(inv: Inventory):
    current_pickaxe = inv.wood_pickaxe + inv.stone_pickaxe + inv.iron_pickaxe 
    # Check if pickaxe can be crafted
    can_craft_wood = inv.wood >= 1
    should_craft_wood = (current_pickaxe < 1) & can_craft_wood
    can_craft_stone = (inv.stone >= 1) & (inv.wood >= 1)
    should_craft_stone = (current_pickaxe < 2) & can_craft_stone
    can_craft_iron = (inv.iron >= 1) & (inv.stone >= 1) & (inv.wood >= 1) & (inv.coal >= 1)
    should_craft_iron = (current_pickaxe < 3) & can_craft_iron

    return (should_craft_wood | should_craft_stone | should_craft_iron ) 

def should_craft_sword(inv: Inventory):
    current_sword = inv.wood_sword + inv.stone_sword + inv.iron_sword
    # Check if sword can be crafted
    can_craft_wood = inv.wood >= 1
    should_craft_wood = (current_sword < 1) & can_craft_wood
    can_craft_stone = (inv.stone >= 1) & (inv.wood >= 1)
    should_craft_stone = (current_sword < 2) & can_craft_stone
    can_craft_iron = (inv.iron >= 1) & (inv.stone >= 1) & (inv.wood >= 1) & (inv.coal >= 1)
    should_craft_iron = (current_sword < 3) & can_craft_iron

    return (should_craft_wood | should_craft_stone | should_craft_iron)

@jax.jit
def crafting_reward(prev_state: CraftaxState, state: CraftaxState):
    # Reward for crafting items (pickaxe1 wood, pickaxe2 stone, ...)
    pickaxe_reward = state.inventory.wood_pickaxe+state.inventory.stone_pickaxe+state.inventory.iron_pickaxe - prev_state.inventory.wood_pickaxe-prev_state.inventory.stone_pickaxe-prev_state.inventory.iron_pickaxe 
    pickaxe_reward = jnp.where(should_craft_pickaxe(state.inventory), pickaxe_reward, 0)
    sword_reward = state.inventory.wood_sword+state.inventory.stone_sword+state.inventory.iron_sword - prev_state.inventory.wood_sword-prev_state.inventory.stone_sword-prev_state.inventory.iron_sword
    sword_reward = jnp.where(should_craft_sword(state.inventory), sword_reward, 0)
    craft_reward = pickaxe_reward + sword_reward 

    return craft_reward

@jax.jit
def explore(prev_state: CraftaxState, state: CraftaxState):
    # Reward movement
    explore_reward = jnp.where(state.player_position[0] != prev_state.player_position[0], 1, 0)
    explore_reward = jnp.where(state.player_position[1] != prev_state.player_position[1], explore_reward, 0)
    return explore_reward 


@partial(jax.jit, static_argnums=(0))
def llm_meta_policy(network, meta_train_state, last_obs, env_state: CraftaxState):
    """
    Mutually exclusive.
    """
    state = env_state
    if isinstance(env_state, MultiRewardLogEnvState): 
        state = env_state.env_state

    # Skill indices mapping
    SKILLS = {
        'SURVIVAL': 0,
        'COMBAT': 1,
        'CRAFT': 2,
        'RESOURCE': 3,
        'EXPLORE': 4,
    }

    # 1. Survival emergencies (health, hunger, thirst, energy)
    survival_mask = (
        (state.player_health < 5) | 
        (state.player_food < 4) | 
        (state.player_drink < 4) | 
        (state.player_energy < 2)
    )

    # 2. Combat priority - check nearby enemies
    player_pos = state.player_position[:, None, :]  # [N, 1, 2]

    zombie_distances = jnp.abs(player_pos - state.zombies.position)  # [N, 2] - [N, M, 2] = [N, M, 2]
    zombie_distances = jnp.linalg.norm(zombie_distances, axis=-1)  # [N, M]
    zombie_distances = jnp.where(state.zombies.mask, zombie_distances, jnp.inf)  # [N, M, 2]
    skeleton_distances = jnp.abs(player_pos - state.skeletons.position)  # [N, 2] - [N, M, 2] = [N, M, 2]
    skeleton_distances = jnp.linalg.norm(skeleton_distances, axis=-1)  # [N, M]
    skeleton_distances = jnp.where(state.skeletons.mask, skeleton_distances, jnp.inf)  # [N, M, 2]
    cow_distances = jnp.abs(player_pos - state.cows.position)  # [N, 2] - [N, M, 2] = [N, M, 2]
    cow_distances = jnp.linalg.norm(cow_distances, axis=-1)  # [N, M]
    cow_distances = jnp.where(state.cows.mask, cow_distances, jnp.inf)  # [N, M, 2]

    combat_mask = jnp.logical_or(jnp.logical_or(jnp.any(zombie_distances <= 3, axis=-1),
                jnp.any(skeleton_distances <= 4, axis=-1)),
                jnp.any(cow_distances <= 2, axis=-1))# [N, M] -> [N]

    # 3. If crafting possible, do it
    
    craft_mask = (should_craft_pickaxe(state.inventory) |
                 should_craft_sword(state.inventory))
    # 3. Resource needs (tools/materials)

    # Should have everything stone in level 1
    # And at least iron in level 3

    # So enforce having at least 5 of stone at lvl 1 
    # Also enforce that lots of wood (20) and coal is collected in lvl 1, since cannot get it later
    # 5 iron and 5 coal at lvl 2 
    # 5 diamond at lvl 5


    inv = state.inventory
    resource_mask = (
        ((inv.stone < 5) | (inv.wood < 5) | (inv.coal < 5)) |
        (state.inventory.stone_pickaxe > 0) & ((inv.iron < 5) | (inv.coal < 5)) |
        (state.inventory.iron_pickaxe > 0) & ((inv.diamond < 1))
    )

    # Priority-based selection using jnp.select
    selection = jnp.select(
        condlist=[
            survival_mask,
            combat_mask,
            craft_mask,
            resource_mask,
        ],
        choicelist=[
            jnp.full(state.player_direction.shape, SKILLS['SURVIVAL']),
            jnp.full(state.player_direction.shape, SKILLS['COMBAT']),
            jnp.full(state.player_direction.shape, SKILLS['CRAFT']),
            jnp.full(state.player_direction.shape, SKILLS['RESOURCE']),
        ],
        # Default: explore the map
        default=jnp.full(state.player_direction.shape, SKILLS['EXPLORE'])
    )
    return jax.nn.one_hot(
        selection,
        len(SKILLS)
    )

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

# def combined_meta_policy_explicit(network, meta_train_state, last_obs, env_state: CraftaxState):
#     llm_q_vals = llm_meta_policy(network, meta_train_state, last_obs, env_state)
#     conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
#     learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
#     combined_q_vals = conditional_q_vals * learned_q_vals
#     return llm_q_vals, combined_q_vals


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
