import jax
import jax.numpy as jnp
from jaxatari.wrappers import MultiRewardLogState, AtariState
from jaxatari.games.jax_kangaroo import KangarooState, JaxKangaroo, get_level_constants

def unpack(state):
    while not isinstance(state, KangarooState):
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        elif hasattr(state, 'env_state'):
            state = state.env_state
        else:
            raise ValueError("State is not a PongState or does not contain a PongState.")
    return state

def env_reward(prev_state: KangarooState, state: KangarooState) -> float:
    """
    Compute the environment reward based on the previous and current state.
    """
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Compute the environment reward based on the previous and current state
    return JaxKangaroo()._get_env_reward(prev_state, state)

@jax.jit
def shaped_reward(prev_state: KangarooState, state: KangarooState):
    # For ablations of PQN with rewardshaping
    # for each leaf in state, add batch_dim 1
    meta_pol_state = jax.tree.map(lambda x: x[None], state)
    choice_qvals = llm_meta_policy(None, None, None, meta_pol_state)
    reward = choice_qvals[:, 0] * navigate_reward(prev_state, state) + choice_qvals[:, 1] * handle_enemies_reward(prev_state, state) + choice_qvals[:, 2] * collect_fruits_reward(prev_state, state)
    return reward.squeeze()

@jax.jit
def shaped_reward_simple(prev_state: KangarooState, state: KangarooState):
    # For ablations of PQN with rewardshaping
    # for each leaf in state, add batch_dim 1
    reward = navigate_reward(prev_state, state) + handle_enemies_reward(prev_state, state) + collect_fruits_reward(prev_state, state)
    return reward.squeeze()

def shaped_reward_llm(prev_state: KangarooState, state: KangarooState):
    prev_obs = JaxKangaroo()._get_observation(prev_state)
    curr_obs = JaxKangaroo()._get_observation(state)
    reward = 0.0

    # 1. Distance-to-Goal Shaping (Potential-based)
    # Calculate Euclidean distance to the baby
    prev_dist = jnp.linalg.norm(prev_obs.player_x - prev_obs.child_position[0]) + \
                jnp.linalg.norm(prev_obs.player_y - prev_obs.child_position[1])
    curr_dist = jnp.linalg.norm(curr_obs.player_x - curr_obs.child_position[0]) + \
                jnp.linalg.norm(curr_obs.player_y - curr_obs.child_position[1])
    
    # Positive reward for getting closer, negative for moving away
    reward += (prev_dist - curr_dist) * 0.1

    # 2. Altitude Reward (Climbing ladders/branches)
    # In Kangaroo, the baby is usually at the top (low Y value in many screen coordinates)
    # We reward decreasing the Y coordinate (moving up)
    # if curr_obs.player_y < prev_obs.player_y:
    #     reward += 0.5
    # elif curr_obs.player_y > prev_obs.player_y:
    #     reward -= 0.2  # Slight penalty for falling/backtracking
    reward = jax.lax.cond(
        curr_obs.player_y < prev_obs.player_y,
        lambda: reward + 0.5,  # if player moved up, add reward
        lambda: jax.lax.cond(
            curr_obs.player_y > prev_obs.player_y,
            lambda: reward - 0.2,  # if player moved down, subtract reward
            lambda: reward  # if no vertical movement, keep reward the same
        )
    )

    # 3. Combat & Threat Neutralization
    # Reward the agent for reducing the number of active monkeys or projectiles in its vicinity
    # (Assuming these disappear from the array when punched)
    prev_monkeys = jnp.count_nonzero(prev_state.level.monkey_states, axis=-1)
    curr_monkeys = jnp.count_nonzero(state.level.monkey_states, axis=-1)
    # if curr_monkeys < prev_monkeys:
    #     reward += 2.0  # Significant reward for punching a monkey
    reward = jax.lax.cond(
        curr_monkeys < prev_monkeys,
        lambda: reward + 2.0,  # if fewer monkeys remain, add reward
        lambda: reward  # otherwise, keep reward the same
    )

    # 4. Collection Logic (Fruit)
    # Small bonus for picking fruit to encourage high-score behavior without distracting from the rescue
    prev_fruits = jnp.count_nonzero(prev_state.level.fruit_actives, axis=-1)
    curr_fruits = jnp.count_nonzero(state.level.fruit_actives, axis=-1)
    # if curr_fruits < prev_fruits:
    #     reward += 1.0
    reward = jax.lax.cond(
        curr_fruits < prev_fruits,
        lambda: reward + 1.0,  # if fewer fruits remain, add reward
        lambda: reward  # otherwise, keep reward the same
    )

    # 5. Living Penalty / Time Pressure
    # Encourages the agent to solve the screen before the Bonus Timer hits zero
    reward -= 0.01

    return reward

@jax.jit
def navigate_reward(prev_state: KangarooState, state: KangarooState):
    # reward for navigating towards the child (upwards)
    reward = jax.lax.cond(
        state.player.y > 160,
        lambda: -0.3,  # if player is below 160, return -0.3
        lambda: jax.lax.cond(
            state.player.is_crashing,
            lambda: 0.,  # if player is crashing, return 0
            lambda: jnp.clip(prev_state.player.y - state.player.y, -9, 9).astype(jnp.float32)  # else return the difference in y position, clipped between -9 and 9
        ),
    )
    return reward 

@jax.jit
def handle_enemies_reward(prev_state: KangarooState, state: KangarooState):
    # reward killing enemies (monkeys) and avoiding death
    state = unpack(state)
    prev_state = unpack(prev_state)
    monkey_reward = jnp.where(jnp.count_nonzero(state.level.monkey_states, axis=-1) < jnp.count_nonzero(prev_state.level.monkey_states, axis=-1), 1, 0)
    dying_reward = jnp.where(state.lives < prev_state.lives, -1, 0)

    reward = monkey_reward + dying_reward
    return reward

@jax.jit
def collect_fruits_reward(prev_state: KangarooState, state: KangarooState):
    # reward for collecting fruits
    state = unpack(state)
    prev_state = unpack(prev_state)
    prev_active = jnp.sum(prev_state.level.fruit_actives, axis=-1)
    new_active = jnp.sum(state.level.fruit_actives, axis=-1)
    reward = jnp.where(new_active < prev_active, 1, 0)
    return reward

# @jax.jit
def llm_meta_policy(network, meta_train_state, last_obs, env_state: KangarooState):
    """
    Mutually exclusive rules.
    """
    state = unpack(env_state)

    # 0: navigate, 1: handle enemies, 2: collect fruits
    
    # default is navigation

    # if fruit or bell is close, collect fruit/activate bell
    max_fruit_dist_sq = 35 ** 2
    fruit_mask = jnp.where(state.level.fruit_actives != 0, 1, 0) #(128, 3)
    print(state.level.fruit_positions.shape, state.player.x.shape)
    dx = state.level.fruit_positions[..., 0] - state.player.x[:, None] #(128, 3)
    dy = state.level.fruit_positions[..., 1] - state.player.y[:, None] #(128, 3)
    fruit_dist_sq = dx ** 2 + dy ** 2
    fruit_close = fruit_dist_sq < max_fruit_dist_sq #(128, 3)
    # mask inactive fruits
    fruit_close = fruit_close * fruit_mask #(128, 3)
    # sum over all fruits
    fruit_close = jnp.sum(fruit_close, axis=1) #(128)

    decision = jnp.where(fruit_close, 2, 0)

    # if enemy is close, handle enemies
    danger_dist_sq = 50 ** 2
    active_mask = jnp.where(state.level.monkey_states != 0, 1, 0) #(128, 4)

    dx = state.level.monkey_positions[..., 0] - state.player.x[:, None] #(128, 4)
    dy = state.level.monkey_positions[..., 1] - state.player.y[:, None]#(128, 4)
    enemy_dist_sq = dx ** 2 + dy ** 2
    enemy_close = enemy_dist_sq < danger_dist_sq #(128, 4)
    # mask inactive enemies
    enemy_close = enemy_close * active_mask #(128, 4)
    # sum over all enemies
    enemy_close = jnp.sum(enemy_close, axis=1) #(128)

    # if coco is close, handle enemies
    # active mask on coco
    coco_falling_mask = jnp.where(state.level.falling_coco_dropping != 0, 1, 0)[:, None] #(128, 5)
    coco_active_mask = jnp.where(state.level.coco_states != 0, 1, 0) #(128, 5)
    coco_mask = jnp.concatenate((coco_falling_mask, coco_active_mask), axis=-1) #(128, 5)
    coco_positions_x = jnp.concatenate((state.level.falling_coco_position[..., :1], state.level.coco_positions[..., 0]), axis=-1) #(128, 5)
    coco_positions_y = jnp.concatenate((state.level.falling_coco_position[..., 1:], state.level.coco_positions[..., 1]), axis=-1) #(128, 5)
    dx = coco_positions_x - state.player.x[:, None] #(128, 5)
    dy = coco_positions_y - state.player.y[:, None] #(128, 5)
    coco_dist_sq = dx ** 2 + dy ** 2
    closest_idx = jnp.argmin(coco_dist_sq, axis=1) #(128,)
    # coco is close
    coco_close = coco_dist_sq[closest_idx] < danger_dist_sq #(128, 5)
    coco_close = coco_close * coco_mask #(128, 5)
    # sum over all coco
    coco_close = jnp.sum(coco_close, axis=1) #(128)


    # possibly overwrite fruit decision 
    decision= jnp.where(jnp.logical_or(enemy_close, coco_close), 1, decision)

    # rewrite decision to 'fake' Q-vals
    q_vals = jax.nn.one_hot(decision, 3)

    return q_vals

def conditional_meta_policy(network, meta_train_state, last_obs, env_state: KangarooState):
    """
    Non mutually exclusive.
    """
    state = unpack(env_state)
    # 0: navigate, 1: handle enemies, 2: collect fruits

    # if fruit or bell is close, collect fruits
    max_fruit_dist_sq = 35 ** 2
    fruit_mask = jnp.where(state.level.fruit_actives != 0, 1, 0) #(128, 3)
    dx = state.level.fruit_positions[..., 0] - state.player.x[:, None] #(128, 3)
    dy = state.level.fruit_positions[..., 1] - state.player.y[:, None] #(128, 3)
    fruit_dist_sq = dx ** 2 + dy ** 2
    fruit_close = fruit_dist_sq < max_fruit_dist_sq #(128, 3)
    # mask inactive fruits
    fruit_close = fruit_close * fruit_mask #(128, 3)
    # sum over all fruits
    fruit_close = jnp.sum(fruit_close, axis=1) #(128)


    bell_mask = jnp.where(state.level.bell_timer == 0, 1, 0) #(128,)
    dx = state.level.bell_position[..., 0] - state.player.x #(128, 1)
    dy = state.level.bell_position[..., 1] - state.player.y #(128, 1)
    bell_dist_sq = dx ** 2 + dy ** 2
    bell_close = bell_dist_sq < max_fruit_dist_sq #(128, 1)
    # mask inactive bells
    bell_close = (bell_close * bell_mask).squeeze() #(128)
    # if fruit or bell is close, collect fruits
    fruit_bell_cond = jnp.logical_or(fruit_close, bell_close) #(128)

    decision = jnp.where(fruit_bell_cond, 2, 0)
    fruit_q = jax.nn.one_hot(decision, 3)

    # if enemy is close, handle enemies
    danger_dist_sq = 50 ** 2
    active_mask = jnp.where(state.level.monkey_states != 0, 1, 0) #(128, 4)

    dx = state.level.monkey_positions[..., 0] - state.player.x[:, None] #(128, 4)
    dy = state.level.monkey_positions[..., 1] - state.player.y[:, None]#(128, 4)
    enemy_dist_sq = dx ** 2 + dy ** 2
    enemy_close = enemy_dist_sq < danger_dist_sq #(128, 4)
    # mask inactive enemies
    enemy_close = enemy_close * active_mask #(128, 4)
    # sum over all enemies
    enemy_close = jnp.sum(enemy_close, axis=1) #(128)

    # if coco is close, handle enemies
    # active mask on coco
    coco_falling_mask = jnp.where(state.level.falling_coco_dropping != 0, 1, 0)[:, None] #(128, 5)
    coco_active_mask = jnp.where(state.level.coco_states != 0, 1, 0) #(128, 5)
    coco_mask = jnp.concatenate((coco_falling_mask, coco_active_mask), axis=-1) #(128, 5)
    coco_positions_x = jnp.concatenate((state.level.falling_coco_position[..., :1], state.level.coco_positions[..., 0]), axis=-1) #(128, 5) coco_positions_y = jnp.concatenate((state.level.falling_coco_position[..., 1:], state.level.coco_positions[..., 1]), axis=-1) #(128, 5)
    coco_positions_y = jnp.concatenate((state.level.falling_coco_position[..., 1:], state.level.coco_positions[..., 1]), axis=-1) #(128, 5)
    dx = coco_positions_x - state.player.x[:, None] #(128, 5)
    dy = coco_positions_y - state.player.y[:, None] #(128, 5)
    coco_dist_sq = dx ** 2 + dy ** 2
    closest_idx = jnp.argmin(coco_dist_sq, axis=1) #(128,)
    # coco is close
    coco_close = coco_dist_sq[closest_idx] < danger_dist_sq #(128, 5)
    coco_close = coco_close * coco_mask #(128, 5)
    # sum over all coco
    coco_close = jnp.sum(coco_close, axis=1) #(128)


    # possibly overwrite fruit decision
    decision = jnp.where(jnp.logical_or(enemy_close, coco_close), 1, 0)
    enemy_q = jax.nn.one_hot(decision, 3)

    # default is navigation (all 0s)
    navigation_q = jnp.zeros_like(fruit_q)

    # rewrite decision to 'fake' Q-vals
    q_vals = jnp.logical_or(navigation_q, fruit_q) 
    q_vals = jnp.logical_or(q_vals, enemy_q)

    return q_vals

def learned_meta_policy(network, meta_train_state, last_obs, env_state: KangarooState):#
    q_vals = network.apply(
        {
            "params": meta_train_state.params,
            "batch_stats": meta_train_state.batch_stats,
        },
        last_obs,
        train=False,
    )
    return q_vals

def combined_meta_policy(network, meta_train_state, last_obs, env_state: KangarooState):
    # combine learned and conditional meta policy
    conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return combined_q_vals

def combined_meta_policy_explicit(network, meta_train_state, last_obs, env_state: KangarooState):
    # combine learned and conditional meta policy
    llm_q_vals = llm_meta_policy(network, meta_train_state, last_obs, env_state)
    conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return llm_q_vals, combined_q_vals

# external rewards for evaluation (not used for training)
def reached_platform_level(prev_state, state) -> jnp.ndarray:
    state = unpack(state)
    prev_state = unpack(prev_state)
    # return +1 for each new platform height reached
    player_bottom_y = state.player.y + state.player.height
    prev_player_bottom_y = prev_state.player.y + prev_state.player.height
    # level_constants = JaxKangaroo()._get_level_constants(state.current_level)
    level_constants = get_level_constants(state.current_level)
    platform_positions_y = level_constants.platform_positions[..., 1]
    filter_first = jnp.where(platform_positions_y >= 172, 0, 1) # bottom platform is at 172
    player_over_platform = player_bottom_y <= platform_positions_y
    prev_player_over_platform = prev_player_bottom_y <= platform_positions_y
    reached_platforms = jnp.sum(jnp.logical_and(
        player_over_platform,
        filter_first,
    ), axis=-1)
    prev_reached_platforms = jnp.sum(jnp.logical_and(
        prev_player_over_platform,
        filter_first,
    ), axis=-1)

    # new reached
    new_reached = jnp.where(reached_platforms > prev_reached_platforms, 1, 0)
    filter_crashed = jnp.logical_or(
        state.player.is_crashing,
        prev_state.player.is_crashing,
    )
    return jnp.where(filter_crashed, 0, new_reached).astype(jnp.float32)

def enemies_killed(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    state = unpack(state)
    prev_state = unpack(prev_state)
    # return +1 for each enemy killed (one fewer monkeys)
    monkey_reward = jnp.where(jnp.count_nonzero(state.level.monkey_states, axis=-1) == jnp.count_nonzero(prev_state.level.monkey_states, axis=-1)-1, 1, 0)
    # filter out player deaths
    filter_crashed = jnp.logical_or(
        state.player.is_crashing,
        prev_state.player.is_crashing,
    )
    return jnp.where(filter_crashed, 0, monkey_reward).astype(jnp.float32)

def fruits_collected(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    state = unpack(state)
    prev_state = unpack(prev_state)
    # return +1 for each fruit collected
    prev_active = jnp.sum(prev_state.level.fruit_actives, axis=-1)
    new_active = jnp.sum(state.level.fruit_actives, axis=-1)
    filter_crashed = jnp.logical_or(
        state.player.is_crashing,
        prev_state.player.is_crashing,
    )
    fruit_collected = jnp.where(new_active < prev_active, 1, 0)
    return jnp.where(filter_crashed, 0, fruit_collected).astype(jnp.float32)