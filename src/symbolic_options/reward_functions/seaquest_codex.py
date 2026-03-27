from jaxatari.games.jax_seaquest import SeaquestState, JaxSeaquest, SeaquestObservation
import jax.numpy as jnp

def unpack(state):
    while not isinstance(state, SeaquestState):
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        elif hasattr(state, 'env_state'):
            state = state.env_state
        else:
            raise ValueError("State is not a SeaquestState or does not contain a SeaquestState.")
    return state

def reward_safe_navigation_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_safe_navigation(prev_obs, curr_obs) 

def reward_collect_divers_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_collect_divers(prev_obs, curr_obs)

def reward_surface_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_surface_with_six(prev_obs, curr_obs)

def reward_oxygen_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_oxygen_management(prev_obs, curr_obs)

def reward_opportunistic_combat_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_opportunistic_combat(prev_obs, curr_obs)

# Assumes SeaquestObservation from your environment.

def _f(x):
    return x.astype(jnp.float32)

def _positive_delta(curr, prev):
    return jnp.maximum(_f(curr) - _f(prev), 0.0)

def _negative_delta(curr, prev):
    return jnp.maximum(_f(prev) - _f(curr), 0.0)

def _active_positions(arr):
    # arr shape: (N, 5) -> x, y, w, h, active
    pos = arr[:, :2]
    active = arr[:, 4] > 0
    return pos, active

def _min_enemy_distance(obs):
    # Requires obs.player.x and obs.player.y in PlayerEntity
    player_xy = jnp.array([_f(obs.player.x), _f(obs.player.y)])

    shark_pos, shark_active = _active_positions(obs.sharks)
    sub_pos, sub_active = _active_positions(obs.submarines)
    missile_pos, missile_active = _active_positions(obs.enemy_missiles)

    all_pos = jnp.concatenate([shark_pos, sub_pos, missile_pos], axis=0)
    all_active = jnp.concatenate([shark_active, sub_active, missile_active], axis=0)

    dists = jnp.linalg.norm(all_pos - player_xy[None, :], axis=1)
    big = jnp.full_like(dists, 1e6)
    dists_masked = jnp.where(all_active, dists, big)
    return jnp.min(dists_masked)

def reward_safe_navigation(prev_obs, curr_obs):
    # Encourage distance from nearby threats + strongly penalize life loss.
    min_dist = _min_enemy_distance(curr_obs)
    danger_radius = 18.0
    proximity_penalty = -0.03 * jnp.maximum(danger_radius - min_dist, 0.0)

    life_lost = _negative_delta(curr_obs.lives, prev_obs.lives)
    death_penalty = -3.0 * life_lost

    alive_bonus = 0.01
    return alive_bonus + proximity_penalty + death_penalty

def reward_collect_divers(prev_obs, curr_obs):
    # Reward picking up divers; penalize losing onboard divers.
    gained_divers = _positive_delta(curr_obs.collected_divers, prev_obs.collected_divers)
    lost_divers = _negative_delta(curr_obs.collected_divers, prev_obs.collected_divers)

    return 1.0 * gained_divers - 0.5 * lost_divers

def reward_surface_with_six(prev_obs, curr_obs):
    # Detect surfacing via oxygen jump (refill event).
    oxygen_jump = _f(curr_obs.oxygen_level) - _f(prev_obs.oxygen_level)
    surfaced = oxygen_jump > 40.0

    had_six = _f(prev_obs.collected_divers) >= 6.0
    now_zero = _f(curr_obs.collected_divers) == 0.0
    successful_delivery = surfaced & had_six & now_zero

    forced_surface = surfaced & (_f(prev_obs.collected_divers) < 6.0)

    # Reward proper cycle completion, penalize early/forced surfacing.
    return 3.0 * _f(successful_delivery) - 1.5 * _f(forced_surface)

def reward_oxygen_management(prev_obs, curr_obs):
    # Penalize staying too long at low oxygen, reward emergency recovery.
    oxy = _f(curr_obs.oxygen_level)
    prev_oxy = _f(prev_obs.oxygen_level)

    low_threshold = 40.0
    low_oxygen_penalty = -0.01 * jnp.maximum(low_threshold - oxy, 0.0)

    surfaced = (oxy - prev_oxy) > 40.0
    emergency_recovery = surfaced & (prev_oxy < low_threshold)

    return low_oxygen_penalty + 0.5 * _f(emergency_recovery)

def reward_opportunistic_combat(prev_obs, curr_obs):
    # Use score increase as sparse proxy for successful combat/events.
    # Strongly penalize deaths so policy stays conservative.
    score_gain = _positive_delta(curr_obs.player_score, prev_obs.player_score)
    life_lost = _negative_delta(curr_obs.lives, prev_obs.lives)

    return 0.003 * score_gain - 2.0 * life_lost
