import jax
import jax.numpy as jnp
from jaxtari.wrappers import MultiRewardLogEnvState, AtariState
from jaxtari.jax_kangaroo import KangarooState

# @jax.jit
# def navigate_reward(prev_state: KangarooState, state: KangarooState):
#     #compute distance to child
#     child_dist = jnp.sqrt((state.player.x - state.level.child_position[..., 0])**2 + (state.player.y - state.level.child_position[..., 1])**2)
#     child_dist_prev = jnp.sqrt((prev_state.player.x - prev_state.level.child_position[..., 0])**2 + (prev_state.player.y - prev_state.level.child_position[..., 1])**2)

#     reward = jnp.where(child_dist < child_dist_prev, 0.1, 0) # reward for getting closer to child
#     reward = jnp.where(child_dist > child_dist_prev, -0.01, reward) # punish for getting further away \
#     # without punishment, the agent would just oscillate 
#     # but because it's okay to move away a bit, the punishment is smaller

#     #TODO: could also add reward for getting closer to the closest ladder


#     # dying punishment
#     reward = jnp.where(state.lives < prev_state.lives, -1, reward)
#     return reward

@jax.jit
def navigate_reward(prev_state: KangarooState, state: KangarooState):
    # navigate to and up ladder
    dx = jnp.abs(state.level.ladder_positions[..., 0] - state.player.x)
    dy = jnp.abs(state.level.ladder_positions[..., 1] - state.player.y)
    dx_prev = jnp.abs(state.level.ladder_positions[..., 0] - prev_state.player.x)

    # find ladder on current level (closest y)
    closest_idx = jnp.argmin(dy) 
    x_diff = dx[closest_idx] - dx_prev[closest_idx]
    ladder_reward = -(x_diff) # reward for getting closer to ladder

    # reward going up ladder
    ladder_up_reward = -(state.player.y - prev_state.player.y)

    dying_reward = jnp.where(state.lives < prev_state.lives, -10, 0)

    return ladder_reward + ladder_up_reward + dying_reward

@jax.jit
def handle_enemies_reward(prev_state: KangarooState, state: KangarooState):
    # threats management
    # reward for killing enemies
    # monkey_state of zero: non-existent
    monkey_reward = jnp.where(jnp.count_nonzero(state.level.monkey_states, axis=-1) < jnp.count_nonzero(prev_state.level.monkey_states, axis=-1), 1, 0)

    # avoid falling/thrown objects
    # coco_positions_x = jnp.concatenate((state.level.falling_coco_position[:1], state.level.morris_coco_positions[:, 0]), axis=-1) #(128, 5)
    # coco_positions_y = jnp.concatenate((state.level.falling_coco_position[1:], state.level.morris_coco_positions[:, 1]), axis=-1) #(128, 5)
    # dx = coco_positions_x - state.player.x
    # dy = coco_positions_y - state.player.y
    # coco_dist_sq = dx ** 2 + dy ** 2
    # closest_idx = jnp.argmin(coco_dist_sq)
    # # reward = distance to closest coco
    # coco_reward = jnp.sqrt(coco_dist_sq[closest_idx]) #(128,)
    # coco_reward /= 10000

    # punish dying
    dying_reward = jnp.where(state.lives < prev_state.lives, -1, 0)
    return monkey_reward + dying_reward

# @jax.jit
# def collect_fruits_reward(prev_state: KangarooState, state: KangarooState):
#     # collecting a fruit: if fewer active than before (they deactivate after collecting)
#     fruit_reward = jnp.where(jnp.count_nonzero(state.level.fruit_actives, axis=-1) < jnp.count_nonzero(prev_state.level.fruit_actives, axis=-1), 1, 0)

#     # reward for ringing bell (player.x and y are the same as bell.x and y; and bell_timer == 0)
#     bell_condition = jnp.logical_and(state.player.x == state.level.bell_position[..., 0], state.player.y == state.level.bell_position[..., 1])
#     bell_condition = jnp.logical_and(bell_condition, state.level.bell_timer == 0)
#     bell_reward = jnp.where(bell_condition, 1, 0) 

#     # punish dying
#     dying_reward = jnp.where(state.lives < prev_state.lives, -10, 0)
#     # combine rewards
#     reward = fruit_reward + bell_reward + dying_reward
#     return reward

@jax.jit
def collect_fruits_reward(prev_state: KangarooState, state: KangarooState):
    # collecting a fruit: if fewer active than before (they deactivate after collecting)
    active_mask = jnp.where(state.level.fruit_actives != 0, 1, 0)

    # compute distance from fruits to player
    dx = jnp.abs(state.level.fruit_positions[..., 0] - state.player.x)
    dy = jnp.abs(state.level.fruit_positions[..., 1] - state.player.y) #(128, 3)
    dx_prev = jnp.abs(state.level.fruit_positions[..., 0] - prev_state.player.x) #(128, 3)
    dy_prev = jnp.abs(state.level.fruit_positions[..., 1] - prev_state.player.y) #(128, 3)
    # set distance of inactive fruits to inf
    dx = jnp.where(state.level.fruit_actives != 0, dx, jnp.inf)
    dy = jnp.where(state.level.fruit_actives != 0, dy, jnp.inf)

    # find closest fruit (closest y)
    closest_idx = jnp.argmin(dy)
    x_diff = dx[closest_idx] - dx_prev[closest_idx]
    y_diff = dy[closest_idx] - dy_prev[closest_idx]
    # reward for getting closer to fruit
    fruit_reward = -((x_diff + y_diff) / 2) # reward for getting closer to fruit
    # if closest fruit is inactive, set reward to 0 (only if all fruits are inactive)
    fruit_reward = jnp.where(state.level.fruit_actives[closest_idx] != 0, fruit_reward, 0)

    # reward for ringing bell (player.x and y are the same as bell.x and y; and bell_timer == 0)
    # bell_condition = jnp.logical_and(state.player.x == state.level.bell_position[..., 0], state.player.y == state.level.bell_position[..., 1])
    # bell_condition = jnp.logical_and(bell_condition, state.level.bell_timer == 0)
    # bell_reward = jnp.where(bell_condition, 10, 0) 
    # reward = -distance to bell
    dx = jnp.abs(state.level.bell_position[..., 0] - state.player.x) #(128, 1)
    dy = jnp.abs(state.level.bell_position[..., 1] - state.player.y)
    # bell_dist_sq = dx ** 2 + dy ** 2
    # bell_dist_sq = jnp.where(state.level.bell_timer == 0, bell_dist_sq, jnp.inf)
    # bell_reward = -jnp.sqrt(bell_dist_sq) #(128,) 
    # bell_reward /= 10000
    # # positive reward
    # bell_reward = 0.01 + bell_reward
    dx_prev = jnp.abs(state.level.bell_position[..., 0] - prev_state.player.x) #(128, 1)
    dy_prev = jnp.abs(state.level.bell_position[..., 1] - prev_state.player.y)

    # set distance of inactive bells to inf
    x_diff = dx - dx_prev
    y_diff = dy - dy_prev
    bell_reward = -((x_diff + y_diff) / 8) # reward for getting closer to bell
    # bell reward should be smaller than fruit reward
    # if closest bell is inactive, set reward to 0 
    bell_reward = jnp.where(state.level.bell_timer == 0, bell_reward, 0)

    # punish dying
    dying_reward = jnp.where(state.lives < prev_state.lives, -30, 0)
    # combine rewards
    reward = fruit_reward + bell_reward + dying_reward
    return reward

# @jax.jit
def llm_meta_policy(network, meta_train_state, last_obs, env_state: KangarooState):
    """
    Mutually exclusive.
    """
    state= env_state
    if isinstance(env_state, MultiRewardLogEnvState) or isinstance(env_state, AtariState):
        state = env_state.env_state
    # 0: navigate, 1: handle enemies, 2: collect fruits
    
    # default is navigation

    # if fruit or bell is close, collect fruits
    max_fruit_dist_sq = 40 ** 2
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

    # if enemy is close, handle enemies
    danger_dist_sq = 35 ** 2
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
    coco_positions_x = jnp.concatenate((state.level.falling_coco_position[..., :1], state.level.morris_coco_positions[..., 0]), axis=-1) #(128, 5)
    coco_positions_y = jnp.concatenate((state.level.falling_coco_position[..., 1:], state.level.morris_coco_positions[..., 1]), axis=-1) #(128, 5)
    dx = coco_positions_x - state.player.x[:, None] #(128, 5)
    dy = coco_positions_y - state.player.y[:, None] #(128, 5)
    coco_dist_sq = dx ** 2 + dy ** 2
    closest_idx = jnp.argmin(coco_dist_sq, axis=1) #(128,)
    # coco is close
    coco_close = coco_dist_sq[closest_idx] < danger_dist_sq #(128, 5)
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
    state= env_state
    if isinstance(env_state, MultiRewardLogEnvState) or isinstance(env_state, AtariState):
        state = env_state.env_state
    # 0: navigate, 1: handle enemies, 2: collect fruits

    # if fruit or bell is close, collect fruits
    max_fruit_dist_sq = 50 ** 2
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
    danger_dist_sq = 45 ** 2
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
    coco_positions_x = jnp.concatenate((state.level.falling_coco_position[..., :1], state.level.morris_coco_positions[..., 0]), axis=-1) #(128, 5)
    coco_positions_y = jnp.concatenate((state.level.falling_coco_position[..., 1:], state.level.morris_coco_positions[..., 1]), axis=-1) #(128, 5)
    dx = coco_positions_x - state.player.x[:, None] #(128, 5)
    dy = coco_positions_y - state.player.y[:, None] #(128, 5)
    coco_dist_sq = dx ** 2 + dy ** 2
    closest_idx = jnp.argmin(coco_dist_sq, axis=1) #(128,)
    # coco is close
    coco_close = coco_dist_sq[closest_idx] < danger_dist_sq #(128, 5)
    # sum over all coco
    coco_close = jnp.sum(coco_close, axis=1) #(128)


    # possibly overwrite fruit decision 
    decision= jnp.where(jnp.logical_or(enemy_close, coco_close), 1, 0)
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