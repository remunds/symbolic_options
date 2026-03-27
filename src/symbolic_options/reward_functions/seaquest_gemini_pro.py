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

def reward_rescue_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_rescue(prev_obs, curr_obs)

def reward_combat_survival_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_combat_survival(prev_obs, curr_obs)

def reward_surface_state(prev_state: SeaquestState, state: SeaquestState):
    prev_state = unpack(prev_state)
    state = unpack(state)

    prev_obs = JaxSeaquest()._get_observation(prev_state)
    curr_obs = JaxSeaquest()._get_observation(state)
    return reward_surface(prev_obs, curr_obs)

def reward_rescue(prev_obs: SeaquestObservation, curr_obs: SeaquestObservation) -> jnp.ndarray:
    """Reward +1.0 for each diver collected."""
    
    # Calculate the difference in collected divers
    delta_divers = curr_obs.collected_divers - prev_obs.collected_divers
    
    # Only reward positive increases (ignores the drop to 0 upon surfacing)
    # Cast boolean to float: True -> 1.0, False -> 0.0
    reward = jnp.where(delta_divers > 0, 1.0, 0.0)
    
    return reward

def reward_combat_survival(prev_obs: SeaquestObservation, curr_obs: SeaquestObservation) -> jnp.ndarray:
    """Reward +1.0 for shooting enemies, Penalty -5.0 for dying."""
    
    delta_score = curr_obs.player_score - prev_obs.player_score
    delta_lives = curr_obs.lives - prev_obs.lives
    
    # Check if a surface event happened (oxygen increased)
    surfaced = curr_obs.oxygen_level > prev_obs.oxygen_level
    
    # Reward score increases ONLY if we are not currently surfacing
    score_reward = jnp.where((delta_score > 0) & (~surfaced), 1.0, 0.0)
    
    # Heavy penalty for losing a life
    death_penalty = jnp.where(delta_lives < 0, -5.0, 0.0)
    
    return score_reward + death_penalty

def reward_surface(prev_obs: SeaquestObservation, curr_obs: SeaquestObservation) -> jnp.ndarray:
    """Reward +2.0 for refilling oxygen, Penalty -5.0 for fatal surfacing mistakes."""
    
    delta_lives = curr_obs.lives - prev_obs.lives
    
    # Check if oxygen replenished
    surfaced = curr_obs.oxygen_level > prev_obs.oxygen_level
    
    # Reward the agent for grabbing a breath of air
    surface_reward = jnp.where(surfaced, 2.0, 0.0)
    
    # Penalize if surfacing resulted in a death (e.g., 0 divers or hitting patrol sub)
    death_penalty = jnp.where(delta_lives < 0, -5.0, 0.0)
    
    return surface_reward + death_penalty