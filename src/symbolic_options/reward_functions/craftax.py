import jax
import jax.numpy as jnp
from craftax.craftax.craftax_state import EnvState as CraftaxState
from symbolic_options.purejaxql.craftax_wrappers import MultiRewardLogEnvState

@jax.jit
def survival_reward(prev_state: CraftaxState, state: CraftaxState):
    # Reward for survival management (player needs)
    health_reward = state.player_health - prev_state.player_health
    food_reward = state.player_food - prev_state.player_food
    drink_reward = state.player_drink - prev_state.player_drink
    energy_reward = state.player_energy - prev_state.player_energy
    mana_reward = state.player_mana - prev_state.player_mana
    return health_reward + food_reward + drink_reward + energy_reward + mana_reward

@jax.jit
def combat_reward(prev_state: CraftaxState, state: CraftaxState):
    # NOTE: this might require additional incentive like rewarding hitting/using bow/...
    # Reward for combat engagement (killing mobs)
    monster_killed_reward = state.monsters_killed[state.player_level] - prev_state.monsters_killed[state.player_level]

    # Punish loss of health
    player_health_reward = state.player_health - prev_state.player_health

    return 5 * monster_killed_reward + player_health_reward

@jax.jit
def resource_collection_reward(prev_state: CraftaxState, state: CraftaxState):
    # Reward for resource collection (wood, stone, coal, iron, diamond)
    wood_reward = state.inventory.wood - prev_state.inventory.wood
    stone_reward = state.inventory.stone - prev_state.inventory.stone
    coal_reward = state.inventory.coal - prev_state.inventory.coal
    iron_reward = state.inventory.iron - prev_state.inventory.iron
    diamond_reward = state.inventory.diamond - prev_state.inventory.diamond
    return wood_reward + 2 * stone_reward + 2 * coal_reward + 3 * iron_reward + 5 * diamond_reward

@jax.jit
def crafting_reward(prev_state: CraftaxState, state: CraftaxState):
    # Reward for crafting items (pickaxe1 wood, pickaxe2 stone, ...)
    pickaxe_reward = state.inventory.pickaxe - prev_state.inventory.pickaxe
    sword_reward = state.inventory.sword - prev_state.inventory.sword
    bow_reward = state.inventory.bow - prev_state.inventory.bow
    armour_reward = jnp.sum(state.inventory.armour - prev_state.inventory.armour, axis=-1)
    torches_reward = state.inventory.torches - prev_state.inventory.torches
    craft_reward = pickaxe_reward + sword_reward + bow_reward + armour_reward + torches_reward

    # Reward for enchanting items (armour, potions)
    sword_enchant_reward = state.sword_enchantment - prev_state.sword_enchantment
    bow_enchant_reward = state.bow_enchantment - prev_state.bow_enchantment
    armour_enchant_reward = jnp.sum(state.armour_enchantments - prev_state.armour_enchantments, axis=-1)
    enchant_reward = sword_enchant_reward + bow_enchant_reward + armour_enchant_reward

    return craft_reward + enchant_reward

@jax.jit
def explore(prev_state: CraftaxState, state: CraftaxState):
    # Reward nothing
    return jnp.zeros_like(state.player_level)

@jax.jit
def level_progression_reward(prev_state: CraftaxState, state: CraftaxState):
    # reward moving closer to the next ladder
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
        'RESOURCE': 2,
        'CRAFT': 3,
        'EXPLORE': 4,
        'LEVEL_PROGRESS': 5
    }

    # 1. Survival emergencies (health, hunger, thirst, energy)
    survival_mask = (
        (state.player_health < 5) | 
        (state.player_food < 3) | 
        (state.player_drink < 3) | 
        (state.player_energy < 3)
    )

    # 2. Combat priority - check nearby enemies
    player_pos = state.player_position[:, None, :]  # [N, 1, 2]

    #mobs_positionl.shape == (N, 9, 3, 2)
    # mobs_mask shape == (N, 9, 3)
    # (N envs, 9 levels, 3 mobs, 2 positions) 
    # player_position.shape == (N, 2)
    # state.player_level.shape == (N, 1)
    # want to get melee_mask with shape (N, 3), where the axis 1 is selected by player_level
    curr_level_melee_mask = state.melee_mobs.mask[jnp.arange(state.melee_mobs.mask.shape[0]), state.player_level]  # [N, 3]
    curr_level_melee_pos = state.melee_mobs.position[jnp.arange(state.melee_mobs.position.shape[0]), state.player_level]  # [N, 3, 2]

    # Check melee mobs
    melee_dists = jnp.sum(
        jnp.abs(curr_level_melee_pos - player_pos), 
        axis=-1
    )  # [N, 3] 
    melee_near = jnp.any(
        (melee_dists <= 2) & curr_level_melee_mask,
        axis=-1
    )  # [N,]

    curr_level_ranged_mask = state.ranged_mobs.mask[jnp.arange(state.ranged_mobs.mask.shape[0]), state.player_level]  # [N, 2]
    curr_level_ranged_pos = state.ranged_mobs.position[jnp.arange(state.ranged_mobs.position.shape[0]), state.player_level]  # [N, 2, 2]

    # Check ranged mobs
    ranged_dists = jnp.sum(
        jnp.abs(curr_level_ranged_pos - player_pos),
        axis=-1
    )  # [N, 2]
    ranged_near = jnp.any(
        (ranged_dists <= 2) & curr_level_ranged_mask, 
        axis=-1
    )  # [N, ]

    combat_mask = melee_near | ranged_near  # [N]

    # 3. Resource needs (tools/materials)

    # Should have everything stone in level 1
    # And at least iron in level 3

    # So enforce having at least 5 of stone at lvl 1 
    # Also enforce that lots of wood (20) and coal is collected in lvl 1, since cannot get it later
    # 5 iron and 5 coal at lvl 2 
    # 5 diamond at lvl 5


    inv = state.inventory
    resource_mask = (
        (state.player_level < 3) & ((inv.stone < 5) | (inv.wood < 20) | (inv.coal < 20)) |
        (state.player_level > 2) & (state.player_level < 5) & ((inv.iron < 5) | (inv.coal < 5)) |
        (state.player_level > 4) & ((inv.diamond < 5))
    )


    # 4. Crafting opportunities

    # player should have everything stone in level 1
    # and at least iron in level 2
    # and diamond in level 5

    craft_mask = (
        (state.player_level < 3) & ((inv.torches < 5) | (inv.pickaxe < 2) | (inv.sword < 2) | (inv.armour < 2).any(axis=-1)) |
        (state.player_level > 2) & (state.player_level < 5) & ((inv.torches < 5) | (inv.pickaxe < 3) | (inv.sword < 3) | (inv.armour < 3).any(axis=-1)) |
        (state.player_level > 4) & ((inv.pickaxe < 4) | (inv.sword < 4) | (inv.armour < 4).any(axis=-1)) 
    )

    # 5. Level progression (current level > 0 and 8 kills)
    level_idx = state.player_level[:, None]  # [N, 1]
    kills = jnp.take_along_axis(
        state.monsters_killed, 
        level_idx, 
        axis=1
    ).squeeze(-1)  # [N]
    level_mask = (state.player_level > 0) & (kills >= 8)  # [N]

    # Priority-based selection using jnp.select
    selection = jnp.select(
        condlist=[
            survival_mask,
            combat_mask,
            resource_mask,
            craft_mask,
            level_mask,
        ],
        choicelist=[
            jnp.full(state.player_level.shape, SKILLS['SURVIVAL']),
            jnp.full(state.player_level.shape, SKILLS['COMBAT']),
            jnp.full(state.player_level.shape, SKILLS['RESOURCE']),
            jnp.full(state.player_level.shape, SKILLS['CRAFT']),
            jnp.full(state.player_level.shape, SKILLS['LEVEL_PROGRESS']),
        ],
        # 6. Default: explore the map
        default=jnp.full(state.player_level.shape, SKILLS['EXPLORE'])
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
