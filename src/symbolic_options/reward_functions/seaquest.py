import jax
import jax.numpy as jnp
from jaxatari.wrappers import MultiRewardLogEnvState, AtariState
from jaxatari.games.jax_seaquest import SeaquestState

@jax.jit
def idle_reward(prev_state: SeaquestState, state: SeaquestState):
    # punish dying give 0.001 else
    reward = jnp.where(state.lives < prev_state.lives, -1, 0.001)
    return reward

@jax.jit
def collect_divers_reward(prev_state: SeaquestState, state: SeaquestState):
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
    # return 1 if player is replenishing oxygen 
    reward = jnp.where(state.oxygen > prev_state.oxygen, 0.1, 0)
    return reward

@jax.jit
def shaped_reward(prev_state: SeaquestState, state: SeaquestState):
    # combine all rewards (+surface with 6 divers reward)
    #TODO: this is a try of balancing the rewards (make collecting divers more valuable than fighting enemies, and encouraging moving up)
    reward = 5 * collect_divers_reward(prev_state, state) + fight_enemies_reward(prev_state, state) + upward_reward(prev_state, state)
    reward = jnp.where(state.successful_rescues > prev_state.successful_rescues, 1000, reward)
    return reward

# @jax.jit
def llm_meta_policy_shoot_default(network, meta_train_state, last_obs, env_state: SeaquestState):
    """
    Mutually exclusive.
    Default is shooting.
    """
    if isinstance(env_state, MultiRewardLogEnvState):
        state = env_state.env_state
    if isinstance(state, AtariState):
        state = state.env_state
    # 0: fight, 1: collect, 2: go up

    # collect (always if divers are present) 
    divers_active = state.diver_positions[..., 2] != 0 # (128, 4)
    divers_per_env_active = jnp.sum(divers_active, axis=1) # (128,)
    decision = jnp.where(divers_per_env_active > 0, 1, 0) 

    # fight (if enemy is close)
    danger_dist_sq = 40 ** 2
    enemy_positions = jnp.concatenate([state.shark_positions, state.sub_positions], axis=1) 
    active_mask = jnp.where(enemy_positions[..., 2] != 0, 1, 0) #(128, 24)

    dx = enemy_positions[..., 0] - state.player_x[:, None] #(128, 24)
    dy = enemy_positions[..., 1] - state.player_y[:, None]#(128, 24)
    enemy_dist_sq = dx ** 2 + dy ** 2
    enemy_close = enemy_dist_sq < danger_dist_sq #(128, 24)
    # mask inactive enemies
    enemy_close = enemy_close * active_mask #(128, 24)
    # sum over all enemies
    enemy_close = jnp.sum(enemy_close, axis=1) #(128)
    decision = jnp.where(enemy_close, 0, decision)

    # go up (if oxygen is low or all divers are collected)
    oxygen_low = state.oxygen < 10
    all_divers_collected = state.divers_collected >= 6
    condition = jnp.logical_or(oxygen_low, all_divers_collected) 
    # possibly override collect and fight decision
    decision = jnp.where(condition, 2, decision)

    # rewrite decision to fake Q-vals
    q_vals = jax.nn.one_hot(decision, 3)

    return q_vals

# @jax.jit
def llm_meta_policy_divers_default(network, meta_train_state, last_obs, env_state: SeaquestState):
    """
    Mutually exclusive. Default is divers.
    """
    if isinstance(env_state, MultiRewardLogEnvState):
        state = env_state.env_state
    if isinstance(state, AtariState):
        state = state.env_state
    # 0: fight, 1: collect, 2: go up

    decision = jnp.ones_like(state.player_x)

    # fight (if enemy is close)
    danger_dist_sq = 40 ** 2
    enemy_positions = jnp.concatenate([state.shark_positions, state.sub_positions], axis=1) 
    active_mask = jnp.where(enemy_positions[..., 2] != 0, 1, 0) #(128, 24)

    dx = enemy_positions[..., 0] - state.player_x[:, None] #(128, 24)
    dy = enemy_positions[..., 1] - state.player_y[:, None]#(128, 24)
    enemy_dist_sq = dx ** 2 + dy ** 2
    enemy_close = enemy_dist_sq < danger_dist_sq #(128, 24)
    # mask inactive enemies
    enemy_close = enemy_close * active_mask #(128, 24)
    # sum over all enemies
    enemy_close = jnp.sum(enemy_close, axis=1) #(128)
    decision = jnp.where(enemy_close, 0, decision)

    # go up (if oxygen is low or all divers are collected)
    oxygen_low = state.oxygen < 10
    all_divers_collected = state.divers_collected >= 6
    condition = jnp.logical_or(oxygen_low, all_divers_collected) 
    # possibly override collect and fight decision
    decision = jnp.where(condition, 2, decision)

    # rewrite decision to fake Q-vals
    q_vals = jax.nn.one_hot(decision, 3)

    return q_vals

def llm_meta_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
    return llm_meta_policy_shoot_default(network, meta_train_state, last_obs, env_state)

# @jax.jit
def divers_default_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
    state = env_state
    #TODO: is there a better way to unpack?
    if isinstance(state, MultiRewardLogEnvState):
        state = state.env_state
    if isinstance(state, AtariState):
        state = state.env_state

    # fight  (if enemy is close)
    danger_dist_sq = 40 ** 2
    enemy_positions = jnp.concatenate([state.shark_positions, state.sub_positions], axis=1) 
    active_mask = jnp.where(enemy_positions[..., 2] != 0, 1, 0) #(128, 24)

    dx = enemy_positions[..., 0] - state.player_x[:, None] #(128, 24)
    dy = enemy_positions[..., 1] - state.player_y[:, None]#(128, 24)
    enemy_dist_sq = dx ** 2 + dy ** 2
    enemy_close = enemy_dist_sq < danger_dist_sq #(128, 24)
    # mask inactive enemies
    enemy_close = enemy_close * active_mask #(128, 24)
    # sum over all enemies
    enemy_close = jnp.sum(enemy_close, axis=1) #(128)
    # NOTE: 1 for collect (default)
    decision= jnp.where(enemy_close, 0, 1)
    enemy_q = jax.nn.one_hot(decision, 3)
    # uses previous decision, and overwrites it with 0 if enemy is close

    # go up (if oxygen is low or all divers are collected)
    oxygen_low = state.oxygen < 10
    all_divers_collected = state.divers_collected >= 6
    condition = jnp.logical_or(oxygen_low, all_divers_collected) 
    # possibly override collect and fight decision
    decision = jnp.where(condition, 2, 1)
    up_q = jax.nn.one_hot(decision, 3)

    q_vals = jnp.logical_or(enemy_q, up_q)

    # add divers_default as always active
    divers_active = jnp.ones_like(decision)
    divers_active = jax.nn.one_hot(divers_active, 3)
    q_vals = jnp.logical_or(q_vals, divers_active) 

    # add some randomness 
    # q_vals = q_vals + jax.random.uniform(jax.random.PRNGKey(0), shape=q_vals.shape) * 0.01
    return q_vals.astype(jnp.float32)

# @jax.jit
def shoot_default_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
    state = env_state
    if isinstance(state, MultiRewardLogEnvState):
        state = state.env_state
    if isinstance(state, AtariState):
        state = state.env_state

    # rescue (always if divers are present)
    divers_active = state.diver_positions[..., 2] != 0 # (128, 4)
    divers_per_env_active = jnp.sum(divers_active, axis=1) # (128,)
    # (1: collect, 0: fight (default))
    decision= jnp.where(divers_per_env_active > 0, 1, 0)
    divers_q = jax.nn.one_hot(decision, 3)


    # go up (if oxygen is low or all divers are collected)
    oxygen_low = state.oxygen < 10
    all_divers_collected = state.divers_collected >= 6
    condition = jnp.logical_or(oxygen_low, all_divers_collected) 

    # (2: go up, 0: fight (default))
    decision = jnp.where(condition, 2, 0)
    up_q = jax.nn.one_hot(decision, 3)

    # fight can always be active
    fight_q = jnp.zeros_like(decision) # always 0 (fight)
    fight_q = jax.nn.one_hot(fight_q, 3)

    q_vals = jnp.logical_or(divers_q, up_q)
    q_vals = jnp.logical_or(q_vals, fight_q)

    # add some randomness 
    # q_vals = q_vals + jax.random.uniform(jax.random.PRNGKey(0), shape=q_vals.shape) * 0.01
    return q_vals


# @jax.jit
def conditional_meta_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
   # choose either divser_default or enemy_default
   return divers_default_policy(network, meta_train_state, last_obs, env_state) 

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