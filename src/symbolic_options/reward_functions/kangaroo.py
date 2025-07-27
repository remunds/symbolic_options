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
# @jax.jit
# def navigate_reward(prev_state: KangarooState, state: KangarooState):
#     # navigate to and up ladder
#     dx = jnp.abs(state.level.ladder_positions[..., 0] - state.player.x)
#     dy = jnp.abs(state.level.ladder_positions[..., 1] - state.player.y)
#     dx_prev = jnp.abs(state.level.ladder_positions[..., 0] - prev_state.player.x)

#     # find ladder on current level (closest y)
#     closest_idx = jnp.argmin(dy) 
#     # x_diff = dx[closest_idx] - dx_prev[closest_idx]
#     ladder_reward = jnp.where(dx[closest_idx] < dx_prev[closest_idx], 1, 0) # reward for getting closer to ladder 
#     ladder_reward = jnp.where(dx[closest_idx] > dx_prev[closest_idx], -1, ladder_reward) # punish for getting further away from ladder 

#     # reward going up (e.g. ladder) 
#     # TODO: before:
#     ladder_up_reward = jnp.where(state.player.y < prev_state.player.y, 1, 0) # reward for going up 
#     ladder_up_reward = jnp.where(state.player.y > prev_state.player.y, -1, ladder_up_reward) # reward for going up 
#     # give only if at ladder
#     ladder_up_reward = jnp.where(dx[closest_idx] < 5, ladder_up_reward, 0)

#     # level up reward
#     level_up_reward = jnp.where(prev_state.current_level != state.current_level, 300, 0)

#     reward = 0.1*ladder_reward + ladder_up_reward + level_up_reward 
#     return reward

@jax.jit
def navigate_reward(prev_state: KangarooState, state: KangarooState):
    # old one that worked
    # reward going up (e.g. ladder) 
    reward = jax.lax.cond(
        state.player.y > 160,
        lambda: -0.3,  # if player is below 160, return -0.1
        lambda: jax.lax.cond(
            state.player.is_crashing,
            lambda: 0.,  # if player is crashing, return 0
            lambda: jnp.clip(prev_state.player.y - state.player.y, -9, 9).astype(jnp.float32)  # else return the difference in y position, clipped between -9 and 9
        ),
    )
    return reward 

# @jax.jit
# def navigate_reward(prev_state, state, reward_state):
#     # reward going up (e.g. ladder) 
#     state = unpack(state)
#     reward = jax.lax.cond(
#         state.player.y < reward_state,
#         lambda: 1, 
#         lambda: 0
#     )
    
#     new_reward_st = jax.lax.cond(
#         state.player.y < reward_state,
#         lambda: state.player.y.astype(jnp.float32),  # if player is below reward_state, return player.y
#         lambda: reward_state  # else return reward_state
#     )

#     return reward, new_reward_st

# @jax.jit
# def navigate_reward(prev_state: KangarooState, state: KangarooState):
#     # navigate to and up ladder
#     dx = jnp.abs(state.level.ladder_positions[..., 0] - state.player.x)
#     dy = jnp.abs(state.level.ladder_positions[..., 1] - state.player.y)
#     dx_prev = jnp.abs(state.level.ladder_positions[..., 0] - prev_state.player.x)
    #     ),
    # )
    # return rewar

# @jax.jit
# def navigate_reward(prev_state: KangarooState, state: KangarooState):
#     # navigate to and up ladder
#     dx = jnp.abs(state.level.ladder_positions[..., 0] - state.player.x)
#     dy = jnp.abs(state.level.ladder_positions[..., 1] - state.player.y)
#     dx_prev = jnp.abs(state.level.ladder_positions[..., 0] - prev_state.player.x)

#     # find ladder on current level (closest y)
#     closest_idx = jnp.argmin(dy) 
#     x_diff = dx[closest_idx] - dx_prev[closest_idx]
#     ladder_reward = -(x_diff) # reward for getting closer to ladder

#     # reward going up (e.g. ladder) 
#     # ladder_up_reward = -3*(state.player.y - prev_state.player.y)
#     # reward reaching new platform
#     ladder_up_reward = reached_platform_level(prev_state, state)*10 
#     # dying_reward = jnp.where(state.lives < prev_state.lives, -10, 0)


#     return ladder_reward + ladder_up_reward #+ dying_reward

# @jax.jit
# def navigate_reward(prev_state: KangarooState, state: KangarooState):
#     # reward going up (e.g. ladder)  - does not work with newest kangaroo version
#     # reward = jax.lax.cond(
#     #     state.player.y > 160,
#     #     lambda: -0.3,  # if player is below 160, return -0.1
#     #     lambda: jax.lax.cond(
#     #         state.player.is_crashing,
#     #         lambda: 0.,  # if player is crashing, return 0
#     #         lambda: jnp.clip(prev_state.player.y - state.player.y, -9, 9).astype(jnp.float32)  # else return the difference in y position, clipped between -9 and 9
#     #     ),
#     # )
#     reward = -(state.player.y - prev_state.player.y)
#     # went 5 down: 45 -> 50 () (50 - 45 = 5, so reward is -5)
#     # went 5 up: 50 -> 45 (45 - 50 = -5, so reward is +5)
#     return reward 

# @jax.jit
# def handle_enemies_reward(prev_state: KangarooState, state: KangarooState):
#     new_crash = jnp.logical_and(
#         state.player.is_crashing,
#         jnp.logical_not(prev_state.player.is_crashing),
#     )
#     return -1 * new_crash.astype(jnp.float32)

@jax.jit
def handle_enemies_reward(prev_state: KangarooState, state: KangarooState):
    state = unpack(state)
    prev_state = unpack(prev_state)
    monkey_reward = jnp.where(jnp.count_nonzero(state.level.monkey_states, axis=-1) < jnp.count_nonzero(prev_state.level.monkey_states, axis=-1), 1, 0)
    # punish dying / reward staying alive
    # alive_reward = jnp.where(state.player.is_crashing, 0, 0.001)
    dying_reward = jnp.where(state.lives < prev_state.lives, -1, 0)

    # level_up_reward = jnp.where(prev_state.current_level != state.current_level, 300, 0)
    reward = monkey_reward + dying_reward #+ alive_reward # + level_up_reward
    return reward

@jax.jit
def collect_fruits_reward(prev_state: KangarooState, state: KangarooState):
    state = unpack(state)
    prev_state = unpack(prev_state)
    # old one that worked
    prev_active = jnp.sum(prev_state.level.fruit_actives, axis=-1)
    new_active = jnp.sum(state.level.fruit_actives, axis=-1)
    reward = jnp.where(new_active < prev_active, 1, 0)
    # problem is: if life lost -> new_active < prev_active 
    # level_up_reward = jnp.where(prev_state.current_level != state.current_level, 300, 0)
    return reward

# Until now: best was just sparse reward, with 0 when crashed.
# @jax.jit
# def collect_fruits_reward(prev_state: KangarooState, state: KangarooState):
#     # sparse reward
#     prev_active = jnp.sum(prev_state.level.fruit_actives, axis=-1)
#     new_active = jnp.sum(state.level.fruit_actives, axis=-1)
#     reward = jnp.where(new_active < prev_active, 100, -0.001)
#     # filter
#     filter_crashed = jnp.logical_or(
#         state.player.is_crashing,
#         prev_state.player.is_crashing,
#     )
#     reward = jnp.where(filter_crashed, -0.01, reward)
#     return reward #+ level_up_reward

# @jax.jit
# def collect_fruits_reward(prev_state, state):
#     # find closest fruit (closest y)
#     dy = jnp.abs(state.level.fruit_positions[..., 1] - state.player.y) #(128, 3)
#     dy = jnp.where(state.level.fruit_actives != 0, dy, jnp.inf)
#     prev_dy = jnp.abs(prev_state.level.fruit_positions[..., 1] - prev_state.player.y) #(128, 3)
#     prev_dy = jnp.where(prev_state.level.fruit_actives != 0, prev_dy, jnp.inf)

#     dx = jnp.abs(state.level.fruit_positions[..., 0] - state.player.x) #(128, 3)
#     dx = jnp.where(state.level.fruit_actives != 0, dx, jnp.inf)
#     prev_dx = jnp.abs(prev_state.level.fruit_positions[..., 0] - prev_state.player.x) #(128, 3)
#     prev_dx = jnp.where(prev_state.level.fruit_actives != 0, prev_dx, jnp.inf)
#     # closest_idx = jnp.argmin(dy)
#     closest_idx = jnp.argmin(prev_dy)
#     diff_x = dx[closest_idx] - prev_dx[closest_idx]
#     diff_y = dy[closest_idx] - prev_dy[closest_idx]
#     # fruit_reward = -(diff_x + diff_y) # reward for getting closer to fruit 
#     fruit_reward = jnp.where(
#         jnp.logical_or(dy[closest_idx] < prev_dy[closest_idx], dx[closest_idx] < prev_dx[closest_idx]),
#         1,  # reward for getting closer to fruit
#         jnp.where(
#             jnp.logical_and(dy[closest_idx] == prev_dy[closest_idx], dx[closest_idx] == prev_dx[closest_idx]),
#             -0.01,  # no change in distance: small penalty
#             -10, # going further away from fruit
#         )
#     )

#     # bonus for actually collecting fruit
#     prev_active = jnp.sum(prev_state.level.fruit_actives, axis=-1)
#     new_active = jnp.sum(state.level.fruit_actives, axis=-1)
#     fruit_reward = jnp.where(new_active < prev_active, 500, fruit_reward)
#     # if player is crashing, return 0
#     filter_crashed = jnp.logical_or(
#         state.player.is_crashing,
#         prev_state.player.is_crashing,
#     )
#     fruit_reward = jnp.where(filter_crashed, -1, fruit_reward)

#     return fruit_reward

# @jax.jit
def llm_meta_policy(network, meta_train_state, last_obs, env_state: KangarooState):
    """
    Mutually exclusive.
    """
    state = unpack(env_state)

    # 0: navigate, 1: handle enemies, 2: collect fruits
    
    # default is navigation

    # if fruit or bell is close, collect fruit/activate bell
    max_fruit_dist_sq = 35 ** 2
    # max_fruit_dist_sq = 35 ** 2
    fruit_mask = jnp.where(state.level.fruit_actives != 0, 1, 0) #(128, 3)
    dx = state.level.fruit_positions[..., 0] - state.player.x[:, None] #(128, 3)
    dy = state.level.fruit_positions[..., 1] - state.player.y[:, None] #(128, 3)
    fruit_dist_sq = dx ** 2 + dy ** 2
    fruit_close = fruit_dist_sq < max_fruit_dist_sq #(128, 3)
    # mask inactive fruits
    fruit_close = fruit_close * fruit_mask #(128, 3)
    # sum over all fruits
    fruit_close = jnp.sum(fruit_close, axis=1) #(128)


    # bell_mask = jnp.where(state.level.bell_timer == 0, 1, 0) #(128,)
    # dx = state.level.bell_position[..., 0] - state.player.x #(128, 1)
    # dy = state.level.bell_position[..., 1] - state.player.y #(128, 1)
    # bell_dist_sq = dx ** 2 + dy ** 2
    # bell_close = bell_dist_sq < max_fruit_dist_sq #(128, 1)
    # # mask inactive bells
    # bell_close = (bell_close * bell_mask).squeeze() #(128)
    # # if fruit or bell is close, collect fruits
    # fruit_bell_cond = jnp.logical_or(fruit_close, bell_close) #(128)

    # decision = jnp.where(fruit_bell_cond, 2, 0)
    decision = jnp.where(fruit_close, 2, 0)

    # if enemy is close, handle enemies
    danger_dist_sq = 50 ** 2
    # danger_dist_sq = 35 ** 2
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

    # rewrite decision to fake Q-vals
    q_vals = jax.nn.one_hot(decision, 3)

    return q_vals

def conditional_meta_policy(network, meta_train_state, last_obs, env_state: KangarooState):
    """
    Non mutually exclusive.
    """
    state = unpack(env_state)
    # state= env_state
    # if isinstance(env_state, MultiRewardLogEnvState):
    #     state = env_state.env_state
    # if isinstance(state, AtariState):
    #     state = state.env_state
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
    # shape (128, 3) -> [[0, 1, 0], [1, 0, 0], ...]
    # fruit_q -> [[0, 0, 1], [1, 0, 0], ...]
    # or -> [[0, 1, 1], [1, 0, 0], ...] (q_weightig decides)
    # default is navigation (all 0s)
    navigation_q = jnp.zeros_like(fruit_q)

    # rewrite decision to fake Q-vals
    # q_vals = jax.nn.one_hot(decision, 3)
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

# external rewards for evaluation

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
    # jax.debug.print("over_platform: {}", player_over_platform) 
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



# GPT generated ones (adapted to our env)

def obstacle_avoidance_reward(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    # reward for avoiding obstacles
    new_crash = jnp.logical_and(
        state.player.is_crashing,
        jnp.logical_not(prev_state.player.is_crashing),
    )
    return jnp.where(new_crash, -1, 0).astype(jnp.float32)

def vertical_navigation_reward(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    # reward for vertical navigation (going up)
    reward = -(state.player.y - prev_state.player.y)
    # if player is below 160, return -0.1
    reward = jax.lax.cond(
        state.player.y > 160,
        lambda: -0.1,
        lambda: reward.astype(jnp.float32),  # if player is below 160, return -0.1
    )
    return reward.astype(jnp.float32)

def fruit_collection_reward(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    # reward for collecting fruits
    prev_active = jnp.sum(prev_state.level.fruit_actives, axis=-1)
    new_active = jnp.sum(state.level.fruit_actives, axis=-1)
    fruit_collected = jnp.where(new_active < prev_active, 1, 0)

    # reward for ringing bell
    bell_rang_cond = jnp.logical_and(prev_state.level.bell_timer == 0, state.level.bell_timer > 0)
    bell_reward = jnp.where(bell_rang_cond, 1, 0)
    fruit_collected += bell_reward
    # if player is crashing, return 0
    filter_crashed = jnp.logical_or(
        state.player.is_crashing,
        prev_state.player.is_crashing,
    )
    return jnp.where(filter_crashed, 0, fruit_collected).astype(jnp.float32)

def goal_reaching_reward(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    # reward for reaching the goal
    goal_reached_cond = jnp.logical_and(state.player.x == state.level.child_position[0], state.player.y == state.level.child_position[1])
    return jnp.where(goal_reached_cond, 1, 0).astype(jnp.float32)