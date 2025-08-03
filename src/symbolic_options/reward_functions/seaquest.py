import jax
import jax.numpy as jnp
from jaxatari.wrappers import AtariState, MultiRewardLogState
from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestState
def unpack(state):
    while not isinstance(state, SeaquestState):
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        elif hasattr(state, 'env_state'):
            state = state.env_state
        else:
            raise ValueError("State is not a SeaquestState or does not contain a SeaquestState.")
    return state

@jax.jit
def env_reward(prev_state: SeaquestState, state: SeaquestState):
    reward = JaxSeaquest()._get_env_reward(prev_state, state)
    return reward

@jax.jit
def collect_divers_reward(prev_state: SeaquestState, state: SeaquestState):
    # return +1 if new diver was collected
    reward = jnp.where(state.divers_collected > prev_state.divers_collected, state.divers_collected, 0)
    return reward

@jax.jit
def fight_enemies_reward(prev_state: SeaquestState, state: SeaquestState):
    # return +1 if an enemy was killed
    # (in this case if the score increased)
    # NOTE: could be adapted (may require adding enemy_kills to state)
    reward = jnp.where(state.score > prev_state.score, 1, 0)
    # dying punishment
    reward = jnp.where(state.lives < prev_state.lives, -1, reward)
    return reward

@jax.jit
def upward_reward(prev_state: SeaquestState, state: SeaquestState):
    # return +0.1 if player is currently replenishing oxygen 
    reward = jnp.where(state.oxygen > prev_state.oxygen, 0.1, 0)
    return reward

@jax.jit
def shaped_reward(prev_state: SeaquestState, state: SeaquestState):
    #NOTE: Not in use in final runs
    # combine all rewards (+surface with 6 divers reward)
    #TODO: this is a try of balancing the rewards (make collecting divers more valuable than fighting enemies, and encouraging moving up)
    reward = 5 * collect_divers_reward(prev_state, state) + fight_enemies_reward(prev_state, state)# + upward_reward(prev_state, state)
    reward = jnp.where(state.successful_rescues > prev_state.successful_rescues, 100, reward)
    return reward

# @jax.jit
def llm_meta_policy_shoot_default(network, meta_train_state, last_obs, env_state: SeaquestState):
    """
    Mutually exclusive.
    Default is shooting.
    """
    state = unpack(env_state)
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
    state = unpack(env_state)
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
    # Using shooting as default policy
    return llm_meta_policy_shoot_default(network, meta_train_state, last_obs, env_state)

# @jax.jit
def divers_default_policy(network, meta_train_state, last_obs, env_state: SeaquestState):
    state = env_state
    state = unpack(state)

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
    state = unpack(state)

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
#    return shoot_default_policy(network, meta_train_state, last_obs, env_state)

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


# External rewards for evaluating alignment with goals (Not used for training)

@jax.jit
def total_rescued(prev_state: SeaquestState, state: SeaquestState):
    # return 1 if player is at surface with 6 divers
    reward = jnp.where(
        state.successful_rescues > prev_state.successful_rescues,
        1.0,
        0.0
    )
    return reward 

@jax.jit
def total_collected(prev_state: SeaquestState, state: SeaquestState):
    # return 1 if player is at surface with 6 divers
    reward = jnp.where(
        state.divers_collected > prev_state.divers_collected,
        1.0,
        0.0
    )
    return reward 

@jax.jit
def total_shot(prev_state: SeaquestState, state: SeaquestState):
    # return 1 if player is at surface with 6 divers
    point_diff = state.score - prev_state.score
    # shark/sub kills are between 20 and 90 points
    # (rescueing is >50*6==300 points)
    enemy_killed = jnp.logical_and(
        point_diff >= 20,
        point_diff <= 90
    )
    reward = jnp.where(
        enemy_killed,
        1.0,
        0.0
    )
    return reward

@jax.jit
def total_surface_without_dying(prev_state, state):
    # return 1 if player reaches surface and doesn't die (at least one diver, no collision, no oxygen_empty)
    at_surface = lambda s: s.player_y <= 47
    close_to_surface = lambda s: s.player_y <= 50
    newly_surfaced = jnp.logical_and(
        at_surface(state),
        jnp.logical_and(
            jnp.logical_not(at_surface(prev_state)),
            close_to_surface(prev_state)
        )
    )
    has_diver = lambda s: s.divers_collected > 0
    oxygen_empty = lambda s: s.oxygen == 0
    surface_cond = jnp.logical_and(
        newly_surfaced,
        jnp.logical_and(
            jnp.logical_not(oxygen_empty(state)), 
            jnp.logical_not(oxygen_empty(prev_state))
        )
    )
    reward = jnp.where(
        jnp.logical_and(
            surface_cond,
            has_diver(state)
        ),
        1.0,
        0.0
    )
    return reward