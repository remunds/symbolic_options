import jax
import jax.numpy as jnp
from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooState


_GAME = JaxKangaroo()


def _unpack(state):
    return state.env_state if hasattr(state, "env_state") else state


# external rewards for evaluation (not used for training)
@jax.jit
def reached_platform_level(prev_state, state) -> jnp.ndarray:
    state = _unpack(state)
    prev_state = _unpack(prev_state)

    player_bottom_y = state.player.y + state.player.height
    prev_player_bottom_y = prev_state.player.y + prev_state.player.height
    level_constants = _GAME._get_level_constants(state.current_level)
    platform_positions_y = level_constants.platform_positions[..., 1]

    # Bottom platform is at y=172, ignore it.
    valid_platform = platform_positions_y < 172
    player_over_platform = player_bottom_y <= platform_positions_y
    prev_player_over_platform = prev_player_bottom_y <= platform_positions_y
    reached_platforms = jnp.sum(
        jnp.logical_and(player_over_platform, valid_platform), axis=-1
    )
    prev_reached_platforms = jnp.sum(
        jnp.logical_and(prev_player_over_platform, valid_platform), axis=-1
    )

    new_reached = jnp.where(reached_platforms > prev_reached_platforms, 1.0, 0.0)
    filter_crashed = jnp.logical_or(
        state.player.is_crashing,
        prev_state.player.is_crashing,
    )
    return jnp.where(filter_crashed, 0.0, new_reached).astype(jnp.float32)


@jax.jit
def enemies_killed(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    state = _unpack(state)
    prev_state = _unpack(prev_state)

    monkey_reward = jnp.where(
        jnp.count_nonzero(state.level.monkey_states, axis=-1)
        == jnp.count_nonzero(prev_state.level.monkey_states, axis=-1) - 1,
        1.0,
        0.0,
    )
    filter_crashed = jnp.logical_or(
        state.player.is_crashing,
        prev_state.player.is_crashing,
    )
    return jnp.where(filter_crashed, 0.0, monkey_reward).astype(jnp.float32)


@jax.jit
def fruits_collected(prev_state: KangarooState, state: KangarooState) -> jnp.ndarray:
    state = _unpack(state)
    prev_state = _unpack(prev_state)

    prev_active = jnp.sum(prev_state.level.fruit_actives, axis=-1)
    new_active = jnp.sum(state.level.fruit_actives, axis=-1)
    fruit_collected = jnp.where(new_active < prev_active, 1.0, 0.0)
    filter_crashed = jnp.logical_or(
        state.player.is_crashing,
        prev_state.player.is_crashing,
    )
    return jnp.where(filter_crashed, 0.0, fruit_collected).astype(jnp.float32)
