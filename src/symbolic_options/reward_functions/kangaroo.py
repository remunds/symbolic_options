import jax
import jax.numpy as jnp
from jaxtari.wrappers import MultiRewardLogEnvState, AtariState
from jaxtari.jax_kangaroo import KangarooState

@jax.jit
def navigate_reward(prev_state: KangarooState, state: KangarooState):
    #compute distance to child
    # prev_child_dist = jnp.sqrt((prev_state.player.x - prev_state.level.child_position[..., 0])**2 + (prev_state.player.y - prev_state.level.child_position[..., 1])**2)

    # child_dist = jnp.sqrt((state.player.x - state.level.child_position[..., 0])**2 + (state.player.y - state.level.child_position[..., 1])**2)
    # distance to child is negative reward
    # reward = -child_dist / 10000

    # reward moving up
    reward = jnp.where(state.player.y < prev_state.player.y, 0.1, -0.1)
    reward = jnp.where(state.player.y == prev_state.player.y, 0.0, reward)

    # reward real progress towards child
    # reward = jnp.where(child_dist < prev_child_dist, 0.01, -0.01)
    # reward = jnp.where(child_dist == prev_child_dist, 0.0, reward) 
    # dying punishment
    reward = jnp.where(state.lives < prev_state.lives, -10, reward)
    return reward

# @jax.jit
# def handle_enemies_reward(prev_state: KangarooState, state: KangarooState):
#     # reward for killing enemies
#     # monkey_state of zero: non-existent
#     reward = jnp.where(jnp.count_nonzero(state.level.monkey_states, axis=-1) < jnp.count_nonzero(prev_state.level.monkey_states, axis=-1), 1, 0)
#     # avoid falling/thrown objects
#     # punish dying
#     reward = jnp.where(state.lives < prev_state.lives, -10, reward)
#     return reward
@jax.jit
def handle_enemies_reward(prev_state: KangarooState, state: KangarooState):
    # reward moving closer to enemy 
    # monkey_state of zero: non-existent
    # compute distance to closest enemy
    dx = state.level.monkey_positions[..., 0] - state.player.x#(128, 4)
    dy = state.level.monkey_positions[..., 1] - state.player.y#(128, 4)
    enemy_dist_sq = dx ** 2 + dy ** 2
    # find closest enemy
    closest_idx = jnp.argmin(enemy_dist_sq)
    # reward = -distance to closest enemy
    enemy_reward = -jnp.sqrt(enemy_dist_sq[closest_idx]) #(128,)
    enemy_reward /= 10000
    # only allow active enemies
    enemy_reward = jnp.where(state.level.monkey_states[closest_idx] != 0, enemy_reward, 0)

    # reward for killing enemies
    kill_reward = jnp.where(jnp.count_nonzero(state.level.monkey_states, axis=-1) < jnp.count_nonzero(prev_state.level.monkey_states, axis=-1), 10, 0)

    # avoid falling/thrown objects
    # punish dying
    dying_reward = jnp.where(state.lives < prev_state.lives, -10, 0)
    # combine rewards
    reward = enemy_reward + kill_reward + dying_reward
    return reward



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
    active_mask = jnp.where(state.level.fruit_actives != 0, 1, 0) #(128, 3)

    # compute distance from fruits to player
    dx = state.level.fruit_positions[..., 0] - state.player.x #(128, 3)
    dy = state.level.fruit_positions[..., 1] - state.player.y #(128, 3)
    # find closest fruit
    fruit_dist_sq = dx ** 2 + dy ** 2
    fruit_dist_sq = jnp.where(active_mask == 0, jnp.inf, fruit_dist_sq) #(128, 3)
    closest_idx = jnp.argmin(fruit_dist_sq)
    
    # reward = -distance to closest fruit
    fruit_reward = -jnp.sqrt(fruit_dist_sq[closest_idx]) #(128,)
    fruit_reward /= 10000
    # only allow active fruits
    fruit_reward = jnp.where(state.level.fruit_actives[closest_idx] != 0, fruit_reward, 0)


    # reward for ringing bell (player.x and y are the same as bell.x and y; and bell_timer == 0)
    # bell_condition = jnp.logical_and(state.player.x == state.level.bell_position[..., 0], state.player.y == state.level.bell_position[..., 1])
    # bell_condition = jnp.logical_and(bell_condition, state.level.bell_timer == 0)
    # bell_reward = jnp.where(bell_condition, 10, 0) 
    # reward = -distance to bell
    dx = state.level.bell_position[..., 0] - state.player.x #(128, 1)
    dy = state.level.bell_position[..., 1] - state.player.y
    bell_dist_sq = dx ** 2 + dy ** 2
    bell_dist_sq = jnp.where(state.level.bell_timer == 0, bell_dist_sq, jnp.inf)
    bell_reward = -jnp.sqrt(bell_dist_sq) #(128,) 
    bell_reward /= 10000
    # only allow active bells
    bell_reward = jnp.where(state.level.bell_timer == 0, bell_reward, 0)

    # punish dying
    dying_reward = jnp.where(state.lives < prev_state.lives, -10, 0)
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
    max_fruit_dist_sq = 45 ** 2
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

    # possibly overwrite fruit decision 
    decision= jnp.where(enemy_close, 1, decision)

    # rewrite decision to fake Q-vals
    q_vals = jax.nn.one_hot(decision, 3)

    return q_vals