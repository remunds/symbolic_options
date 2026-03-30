import jax
import jax.numpy as jnp


@jax.jit
def total_rescued(prev_state, state):
    # return 1 if a rescue count increment happened
    reward = jnp.where(
        state.successful_rescues > prev_state.successful_rescues,
        1.0,
        0.0
    )
    return reward


@jax.jit
def total_collected(prev_state, state):
    # return 1 if a diver collection increment happened
    reward = jnp.where(
        state.divers_collected > prev_state.divers_collected,
        1.0,
        0.0
    )
    return reward


@jax.jit
def total_shot(prev_state, state):
    # return 1 when score jump suggests an enemy kill
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