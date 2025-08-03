from functools import partial
import jax
import jax.numpy as jnp
from craftax.craftax_classic.envs.craftax_state import EnvState as CraftaxState, Inventory
from symbolic_options.purejaxql.craftax_wrappers import MultiRewardLogEnvState

def unpack(state):
    while not isinstance(state, CraftaxState):
            state = state.env_state
    return state

def env_reward(prev_state: CraftaxState, state: CraftaxState) -> float:
    """
    Compute the environment reward based on the previous and current state.
    """
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Compute the environment reward based on the previous and current state
    achievement_reward = (
        prev_state.achievements.astype(jnp.float32).sum()
        - state.achievements.astype(jnp.float32).sum() 
    )
    health_reward = (prev_state.player_health - state.player_health) * 0.1
    reward = achievement_reward + health_reward
    return reward 

@jax.jit
def survival_reward(prev_state: CraftaxState, state: CraftaxState):
    prev_state = unpack(prev_state)
    state = unpack(state)
    # this works great
    # Reward for survival management (player needs)
    health_reward = state.player_health - prev_state.player_health
    food_reward = state.player_food - prev_state.player_food
    drink_reward = state.player_drink - prev_state.player_drink
    energy_reward = state.player_energy - prev_state.player_energy
    return 5*health_reward + 2*food_reward + 2*drink_reward + 0.5*energy_reward

@jax.jit
def combat_reward(prev_state: CraftaxState, state: CraftaxState):
    prev_state = unpack(prev_state)
    state = unpack(state)
    # this works great
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
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Reward for resource collection (wood, stone, coal, iron, diamond)
    # Limit the rewards based on the already collected resources 
    wood_reward = state.inventory.wood - prev_state.inventory.wood
    wood_reward = jnp.where(prev_state.inventory.wood < 5, wood_reward, 0)
    stone_reward = state.inventory.stone - prev_state.inventory.stone
    stone_reward = jnp.where(prev_state.inventory.stone < 5, stone_reward, 0)
    coal_reward = state.inventory.coal - prev_state.inventory.coal
    coal_reward = jnp.where(prev_state.inventory.coal < 5, coal_reward, 0)
    iron_reward = state.inventory.iron - prev_state.inventory.iron
    iron_reward = jnp.where(prev_state.inventory.iron < 5, iron_reward, 0)
    diamond_reward = state.inventory.diamond - prev_state.inventory.diamond
    resource_reward = wood_reward + 5 * stone_reward + 3 * coal_reward + 10 * iron_reward + 20 * diamond_reward
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

    return [should_craft_wood, should_craft_stone, should_craft_iron]

def should_craft_sword(inv: Inventory):
    current_sword = inv.wood_sword + inv.stone_sword + inv.iron_sword
    # Check if sword can be crafted
    can_craft_wood = inv.wood >= 1
    should_craft_wood = (current_sword < 1) & can_craft_wood
    can_craft_stone = (inv.stone >= 1) & (inv.wood >= 1)
    should_craft_stone = (current_sword < 2) & can_craft_stone
    can_craft_iron = (inv.iron >= 1) & (inv.stone >= 1) & (inv.wood >= 1) & (inv.coal >= 1)
    should_craft_iron = (current_sword < 3) & can_craft_iron

    return [should_craft_wood, should_craft_stone, should_craft_iron]

@jax.jit
def crafting_reward(prev_state: CraftaxState, state: CraftaxState):
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Reward for crafting items (pickaxe1 wood, pickaxe2 stone, ...)
    wood_pick_rew = jnp.where(prev_state.inventory.wood_pickaxe < state.inventory.wood_pickaxe, 1, 0)
    wood_pick_rew = jnp.where(should_craft_pickaxe(prev_state.inventory)[0], wood_pick_rew, 0)
    stone_pick_rew = jnp.where(prev_state.inventory.stone_pickaxe < state.inventory.stone_pickaxe, 1, 0)
    stone_pick_rew = jnp.where(should_craft_pickaxe(prev_state.inventory)[1], stone_pick_rew, 0)
    iron_pick_rew = jnp.where(prev_state.inventory.iron_pickaxe < state.inventory.iron_pickaxe, 1, 0)
    iron_pick_rew = jnp.where(should_craft_pickaxe(prev_state.inventory)[2], iron_pick_rew, 0)
    pickaxe_reward = wood_pick_rew + 2*stone_pick_rew + 5*iron_pick_rew

    wood_sword = jnp.where(prev_state.inventory.wood_sword < state.inventory.wood_sword, 1, 0)
    wood_sword = jnp.where(should_craft_sword(prev_state.inventory)[0], wood_sword, 0)
    stone_sword = jnp.where(prev_state.inventory.stone_sword < state.inventory.stone_sword, 1, 0)
    stone_sword = jnp.where(should_craft_sword(prev_state.inventory)[1], stone_sword, 0)
    iron_sword = jnp.where(prev_state.inventory.iron_sword < state.inventory.iron_sword, 1, 0)
    iron_sword = jnp.where(should_craft_sword(prev_state.inventory)[2], iron_sword, 0)
    sword_reward = wood_sword + 2*stone_sword + 5*iron_sword 

    craft_reward = pickaxe_reward + sword_reward 

    return craft_reward

@jax.jit
def explore(prev_state: CraftaxState, state: CraftaxState):
    # Reward reaching unexplored areas
    # Note: this uses added exploration map
    new_explored = state.exploration_map.astype(int) - prev_state.exploration_map.astype(int)
    new_explored = jnp.where(new_explored > 0, 0.05, 0)
    reward = jnp.sum(new_explored, axis=(-2, -1))  # Sum over the map dimensions
    return reward


@partial(jax.jit, static_argnums=(0))
def llm_meta_policy(network, meta_train_state, last_obs, env_state: CraftaxState):
    """
    Mutually exclusive.
    """
    state = unpack(env_state)
    # if isinstance(env_state, MultiRewardLogEnvState): 
    #     state = env_state.env_state

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
    
    craft_mask = (jnp.stack(should_craft_pickaxe(state.inventory), axis=-1).any(axis=-1) |
                 jnp.stack(should_craft_sword(state.inventory), axis=-1).any(axis=-1))
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
            combat_mask,
            survival_mask,
            craft_mask,
            resource_mask,
        ],
        choicelist=[
            jnp.full(state.player_direction.shape, SKILLS['COMBAT']),
            jnp.full(state.player_direction.shape, SKILLS['SURVIVAL']),
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
    
    craft_mask = (jnp.stack(should_craft_pickaxe(state.inventory), axis=-1).any(axis=-1) |
                 jnp.stack(should_craft_sword(state.inventory), axis=-1).any(axis=-1))
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

    # exploration is always possible
    explore_mask = jnp.full_like(combat_mask, SKILLS['EXPLORE']).astype(int)
    explore_mask = jax.nn.one_hot(explore_mask, len(SKILLS))

    combat_mask = jax.nn.one_hot(
        jnp.where(combat_mask.astype(int), SKILLS['COMBAT'], SKILLS['EXPLORE']),
        len(SKILLS)
    )
    survival_mask = jax.nn.one_hot(
        jnp.where(survival_mask.astype(int), SKILLS['SURVIVAL'], SKILLS['EXPLORE']),
        len(SKILLS)
    )
    craft_mask = jax.nn.one_hot(
        jnp.where(craft_mask.astype(int), SKILLS['CRAFT'], SKILLS['EXPLORE']),
        len(SKILLS)
    )
    resource_mask = jax.nn.one_hot(
        jnp.where(resource_mask.astype(int), SKILLS['RESOURCE'], SKILLS['EXPLORE']),
        len(SKILLS)
    )
    # Combine all masks
    combined_mask = (
        combat_mask + survival_mask + craft_mask + resource_mask + explore_mask
    ) # [N, 5]

    return combined_mask


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
    conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return combined_q_vals