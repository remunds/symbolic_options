import jax
import jax.numpy as jnp
from symbolic_options.wrappers import MultiRewardLogEnvState
from jaxtari.jax_seaquest import SeaquestState

@jax.jit
def collect_divers_reward(prev_state: SeaquestState, state: SeaquestState):
    # return 1 if a new diver was collected
    reward = jnp.where(state.divers_collected > prev_state.divers_collected, 1, 0)
    # dying punishment
    reward = jnp.where(state.lives < prev_state.lives, -1, reward)
    return reward

@jax.jit
def fight_enemies_reward(prev_state: SeaquestState, state: SeaquestState):
    # return 1 if an enemy was killed
    # (in this case if the score increased)
    # NOTE: could be adapted (may require adding enemy_kills to state)
    reward = jnp.where(state.score > prev_state.score, 1, 0)
    # dying punishment
    reward = jnp.where(state.lives < prev_state.lives, -1, reward)
    return reward


@jax.jit
def upward_reward(prev_state: SeaquestState, state: SeaquestState):
    # return 1 if player is moving up 
    reward = jnp.where(state.player_y == prev_state.player_y-1, 0.1, 0)
    # dying punishment
    reward = jnp.where(state.lives < prev_state.lives, -1, reward)
    return reward

@jax.jit
def shaped_reward(prev_state: SeaquestState, state: SeaquestState):
    # combine all rewards (+surface with 6 divers reward)
    #TODO: this is a try of balancing the rewards (make collecting divers more valuable than fighting enemies, and encouraging moving up)
    reward = 5 * collect_divers_reward(prev_state, state) + fight_enemies_reward(prev_state, state) + 10*upward_reward(prev_state, state)
    reward = jnp.where(state.successful_rescues > prev_state.successful_rescues, 100, reward)
    return reward

@jax.jit
def llm_meta_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
    """
    Mutually exclusive.
    """
    if isinstance(env_state, MultiRewardLogEnvState):
        state = env_state.env_state
    # 0: fight, 1: collect, 2: go up

    # smartly choose between the three reward functions
    # based on the state
    # each state array has shape (n_envs, ...)

    # if enemy is close -> fight
    # if diver is available -> collect
    # if collected_divers==6 or oxygen_low -> go up
    # else -> fight (default)
    # also: fight has priority over collect
    # and: go up has highest priority

    # collect 
    divers_active = state.diver_positions[..., 2] != 0 # (128, 4)
    divers_per_env_active = jnp.sum(divers_active, axis=1) # (128,)
    decision = jnp.where(divers_per_env_active > 0, 1, 0) 

    # fight 
    danger_dist_sq = 40 ** 2
    enemy_positions = jnp.concatenate([state.shark_positions, state.sub_positions], axis=1) 
    active_mask = jnp.where(enemy_positions[..., 2] != 0, 1, 0) #(128, 24)

    # In the end, I want to have an array of shape (128) with True if an enemy is close in current env
    # For that, I need to calculate the distance between the player and each enemy
    # and check if it is smaller than danger_dist_sq
    dx = enemy_positions[..., 0] - state.player_x[:, None] #(128, 24)
    dy = enemy_positions[..., 1] - state.player_y[:, None]#(128, 24)
    enemy_dist_sq = dx ** 2 + dy ** 2
    enemy_close = enemy_dist_sq < danger_dist_sq #(128, 24)
    # mask inactive enemies
    enemy_close = enemy_close * active_mask #(128, 24)
    # sum over all enemies
    enemy_close = jnp.sum(enemy_close, axis=1) #(128)
    decision = jnp.where(enemy_close, 0, decision)

    # possibly override collect decision
    decision = jnp.where(enemy_close, 0, decision)

    oxygen_low = state.oxygen < 10
    all_divers_collected = state.divers_collected >= 6
    condition = jnp.logical_or(oxygen_low, all_divers_collected) 
    # possibly override collect and fight decision
    decision = jnp.where(condition, 2, decision)

    # rewrite decision to fake Q-vals
    q_vals = jax.nn.one_hot(decision, 3)

    return q_vals

# @jax.jit
def conditional_meta_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
    """
    Same as llm_meta_policy but not mutually exclusive. 
    """
    if isinstance(env_state, MultiRewardLogEnvState):
        state = env_state.env_state

    # collect 
    divers_active = state.diver_positions[..., 2] != 0 # (128, 4)
    divers_per_env_active = jnp.sum(divers_active, axis=1) # (128,)
    # decision= jnp.where(divers_per_env_active > 0, 1, 0)
    #TODO: this was just a quick test if it's better to always activate divers instead of fighting
    decision= jnp.where(divers_per_env_active > 0, 1, 1)
    divers_q = jax.nn.one_hot(decision, 3)

    # fight 
    danger_dist_sq = 40 ** 2
    enemy_positions = jnp.concatenate([state.shark_positions, state.sub_positions], axis=1) 
    active_mask = jnp.where(enemy_positions[..., 2] != 0, 1, 0) #(128, 24)

    # In the end, I want to have an array of shape (128) with True if an enemy is close in current env
    # For that, I need to calculate the distance between the player and each enemy
    # and check if it is smaller than danger_dist_sq
    dx = enemy_positions[..., 0] - state.player_x[:, None] #(128, 24)
    dy = enemy_positions[..., 1] - state.player_y[:, None]#(128, 24)
    enemy_dist_sq = dx ** 2 + dy ** 2
    enemy_close = enemy_dist_sq < danger_dist_sq #(128, 24)
    # mask inactive enemies
    enemy_close = enemy_close * active_mask #(128, 24)
    # sum over all enemies
    enemy_close = jnp.sum(enemy_close, axis=1) #(128)
    decision= jnp.where(enemy_close, 0, decision)
    enemy_q = jax.nn.one_hot(decision, 3)

    oxygen_low = state.oxygen < 10
    all_divers_collected = state.divers_collected >= 6
    condition = jnp.logical_or(oxygen_low, all_divers_collected) 
    # possibly override collect and fight decision
    decision = jnp.where(condition, 2, decision)
    up_q = jax.nn.one_hot(decision, 3)

    # take logical_or of all conditions to generate fake Q-values
    # [0,0,1] or [1,0,0] -> [1,0,1]
    q_vals = jnp.logical_or(divers_q, enemy_q)
    q_vals = jnp.logical_or(q_vals, up_q)

    # add some randomness 
    q_vals = q_vals + jax.random.uniform(jax.random.PRNGKey(0), shape=q_vals.shape) * 0.01
    return q_vals


#TODO: for typing, we may want to define CustomTrainSeaquestState here (or somewhere common) and import
# @jax.jit
def learned_meta_policy(network, meta_train_state, last_obs, env_state: SeaquestState):#
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
def combined_meta_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
    # combine learned and conditional meta policy
    conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return combined_q_vals