import jax.numpy as jnp
from jax import lax
import jax
from enum import IntEnum
from jaxatari.games.jax_breakout import BreakoutState, BreakoutConstants

def unpack(state):
    while not isinstance(state, BreakoutState):
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        elif hasattr(state, 'env_state'):
            state = state.env_state
        else:
            raise ValueError("State is not a BreakoutState or does not contain a BreakoutState.")
    return state

def env_reward(prev_state: BreakoutState, state: BreakoutState) -> float:
    """
    Compute the environment reward based on the previous and current state.
    """
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Compute the environment reward based on the previous and current state
    return state.score - prev_state.score

def track_and_align(prev: BreakoutState, curr: BreakoutState) -> float:
    """
    Reward ∈ [0, 1] based on reduction in horizontal distance to ball.
    1 when distance significantly reduced, 0 when increased.
    """
    player_width = jnp.where(
        curr.small_paddle,
        BreakoutConstants().PLAYER_SIZE_SMALL[0],
        BreakoutConstants().PLAYER_SIZE[0]
    )

    distance = abs((curr.player_x + player_width / 2) - curr.ball_x)
    reward = jnp.where(distance < 3, 1.0, 0.0)
    return reward

def return_shot(prev: BreakoutState, curr: BreakoutState) -> float:
    """
    This is aiming now instead. -> Just give it env_return s.t. it learns to aim for highest reward.
    """
    env_reward = curr.score - prev.score 

    return env_reward

def defensive_positioning(prev: BreakoutState, curr: BreakoutState) -> float:
    """
    Reward ∈ [0, 1] based on movement toward screen horizontal center.
    1 for maximal improvement, 0 if worsened.
    """
    player_width = jnp.where(
        curr.small_paddle,
        BreakoutConstants().PLAYER_SIZE_SMALL[0],
        BreakoutConstants().PLAYER_SIZE[0]
    )
    screen_center_x = 80+8 #middle = WINDOW_WIDTH // 2 + WALL_WIDTH
    # distance to mid
    dist = abs((curr.player_x + player_width / 2) - screen_center_x)
    reward = jnp.where(dist < 3, 1.0, 0.0)
    return reward

# === 3. Meta-policy Function ===

def llm_meta_policy(network, meta_train_state, last_obs, env_state):
    """
    Decide which skill to use based on ball dynamics and position.
    Priority: Interception > Tracking > Recovery
    """
    state = unpack(env_state)
    player_width = jnp.where(
        state.small_paddle,
        BreakoutConstants().PLAYER_SIZE_SMALL[0],
        BreakoutConstants().PLAYER_SIZE[0]
    )

    # ball is approaching, if ball_vel_y is positive and passed the middle of the screen
    ball_approaching = jnp.logical_and(state.ball_vel_y > 0, state.ball_y > 122) 

    close_to_paddle = abs(state.ball_y - BreakoutConstants().PLAYER_START_Y) < 2*player_width

    # if ball is approaching (after mid) -> track ball
    # if close to paddle -> return ball
    # else (ball not approaching) -> recover to center

    qvals = jnp.where(
        ball_approaching,
        0,  # trackBall
        jnp.where(close_to_paddle, 1, 2)  # ReturnBall or RecoverToCenter
    )
    qvals = jax.nn.one_hot(qvals, 3)  # Assuming 3 skills: TrackBall, InterceptBall, RecoverToCenter
    return qvals.astype(jnp.float32)  # Ensure the output is float32 for consistency

def conditional_meta_policy(network, meta_train_state, last_obs, env_state):
    # default to tracking (always active) 
    # Same as llm_meta_policy, but allow multiple skills to be active at once 
    state = unpack(env_state)
    player_width = jnp.where(
        state.small_paddle,
        BreakoutConstants().PLAYER_SIZE_SMALL[0],
        BreakoutConstants().PLAYER_SIZE[0]
    )
    # ball is approaching, if ball_vel_y is positive and passed the middle of the screen
    ball_approaching = jnp.logical_and(state.ball_vel_y > 0, state.ball_y > 122)
    close_to_paddle = abs(state.ball_y - BreakoutConstants().PLAYER_START_Y) < 2*player_width
    # if ball is approaching (after mid) -> track ball
    # if close to paddle -> return ball
    # else (ball not approaching) -> recover to center
    returnball_qvals = jnp.where(close_to_paddle, 1.0, 0.0)  # ReturnBall
    returnball_qvals = jax.nn.one_hot(returnball_qvals, 3)  # ReturnBall skill
    recover_qvals = jnp.where(~ball_approaching, 2.0, 0.0)  # RecoverToCenter
    recover_qvals = jax.nn.one_hot(recover_qvals, 3)  # RecoverToCenter skill
    track_qvals = jnp.where(ball_approaching, 0.0, 0.0)  # TrackBall, always active
    track_qvals = jax.nn.one_hot(track_qvals, 3)  # TrackBall skill

    # Combine the Q-values
    q_vals = jnp.logical_or(
        jnp.logical_or(returnball_qvals, recover_qvals),
        track_qvals
    )
    return q_vals.astype(jnp.float32)  # Avoid crash or go forward, both can be active at once


def learned_meta_policy(network, meta_train_state, last_obs, env_state):
    q_vals = network.apply(
        {
            "params": meta_train_state.params,
            "batch_stats": meta_train_state.batch_stats,
        },
        last_obs,
        train=False,
    )
    return q_vals.astype(jnp.float32)  # Ensure the output is float32 for consistency

def combined_meta_policy(network, meta_train_state, last_obs, env_state):
    # combine learned and conditional meta policy
    conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return combined_q_vals