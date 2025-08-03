import jax.numpy as jnp
from jax import lax
import jax
from enum import IntEnum
from jaxatari.games.jax_pong import PongState, PongConstants, JaxPong

GAMMA = 0.99 

def unpack(state):
    while not isinstance(state, PongState):
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        elif hasattr(state, 'env_state'):
            state = state.env_state
        else:
            raise ValueError("State is not a PongState or does not contain a PongState.")
    return state

def env_reward(prev_state: PongState, state: PongState) -> float:
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Compute the environment reward based on the previous and current state
    return JaxPong()._get_reward(prev_state, state)


# === Updated Reward Functions ===
PLAYER_X = 140
PLAYER_W = 4
PLAYER_H = 16
SCREEN_CENTER_X = 78
SCREEN_CENTER_Y = 115

def track_and_align(prev: PongState, curr: PongState) -> float:
    """
    Reward ∈ [0, 1] based on reduction in vertical distance to ball.
    1 when distance significantly reduced, 0 when increased.
    """
    env_reward = JaxPong()._get_reward(prev, curr)
    prev = unpack(prev)
    curr = unpack(curr)

    distance = abs((curr.player_y + PLAYER_H / 2) - curr.ball_y)
    reward = jnp.where(distance < 3, 1.0, 0.0)
    return env_reward + reward * 0.1 # scale down

def return_shot(prev: PongState, curr: PongState) -> float:
    """
    Binary reward ∈ {0, 1} if paddle and ball align closely at contact range.
    """
    prev = unpack(prev)
    curr = unpack(curr)
    env_reward = JaxPong()._get_reward(prev, curr)
    x_close = abs(curr.ball_x - PLAYER_X) < 1.5 * PLAYER_W 
    y_align = abs(curr.ball_y - curr.player_y) < 0.5 * PLAYER_H
    ball_speed = abs(curr.ball_x - prev.ball_x) * 0.1
    reward = jnp.where(x_close & y_align, 1.0 + ball_speed, 0.0)

    return env_reward + reward * 0.1

def defensive_positioning(prev: PongState, curr: PongState) -> float:
    """
    Reward ∈ [0, 1] based on movement toward screen vertical center.
    1 for maximal improvement, 0 if worsened.
    """
    prev = unpack(prev)
    curr = unpack(curr)
    screen_center_y = SCREEN_CENTER_Y 
    env_reward = JaxPong()._get_reward(prev, curr)
    # distance to mid
    dist = abs((curr.player_y + PLAYER_H / 2) - screen_center_y)
    reward = jnp.where(dist < 3, 1.0, 0.0)
    return env_reward + reward * 0.1  # scale down 

# === 3. Meta-policy Function ===

def llm_meta_policy(network, meta_train_state, last_obs, env_state):
    """
    Decide which skill to use based on ball dynamics and position.
    Priority: Interception > Tracking > Recovery
    """
    state = unpack(env_state)
    ball_approaching = state.ball_x > SCREEN_CENTER_X  # assume player is on right side
    close_to_paddle = abs(state.ball_x - PLAYER_X) < 5 * PLAYER_W 
    y_misaligned = abs(state.ball_y - (state.player_y - PLAYER_H / 2)) > 0.5 * PLAYER_H
    near_center = abs((state.player_y - PLAYER_H / 2) - SCREEN_CENTER_Y) < 3

    qvals = jnp.where(
        jnp.logical_and(jnp.logical_and(ball_approaching, close_to_paddle), y_misaligned),
        1,  # InterceptBall
        jnp.where(ball_approaching, 0,  # TrackBall
                  jnp.where(~near_center, 2, 0))  # RecoverToCenter
    )
    qvals = jax.nn.one_hot(qvals, 3)  # Assuming 3 skills: TrackBall, InterceptBall, RecoverToCenter
    return qvals 

def conditional_meta_policy(network, meta_train_state, last_obs, env_state):
    # default to tracking (always active) 
    env_state = unpack(env_state)
    ball_approaching = env_state.ball_x > SCREEN_CENTER_X
    close_to_paddle = abs(env_state.ball_x - PLAYER_X) < 10
    y_misaligned = abs(env_state.ball_y - env_state.player_y) > 0.5 * PLAYER_H
    near_center = abs(env_state.player_y - SCREEN_CENTER_Y) < 3

    intercept_cond = jnp.logical_and(
        jnp.logical_and(ball_approaching, close_to_paddle), y_misaligned
    )
    intercept_qvals = jax.nn.one_hot(
        jnp.where(intercept_cond, 1, 0), 3
    )  # InterceptBall

    recover_cond = jnp.logical_and(
        ~ball_approaching, ~near_center
    )
    recover_qvals = jax.nn.one_hot(
        jnp.where(recover_cond, 2, 0), 3
    )  # RecoverToCenter

    track_qvals = jax.nn.one_hot(
        jnp.where(ball_approaching, 0, 0), 3
    )  # TrackBall (always active)
    # Combine the Q-values
    q_vals = jnp.logical_or(
        jnp.logical_or(intercept_qvals, recover_qvals),
        track_qvals
    )
    return q_vals


def learned_meta_policy(network, meta_train_state, last_obs, env_state):
    q_vals = network.apply(
        {
            "params": meta_train_state.params,
            "batch_stats": meta_train_state.batch_stats,
        },
        last_obs,
        train=False,
    )
    return q_vals

def combined_meta_policy(network, meta_train_state, last_obs, env_state):
    # combine learned and conditional meta policy
    conditional_q_vals = conditional_meta_policy(network, meta_train_state, last_obs, env_state)
    learned_q_vals = learned_meta_policy(network, meta_train_state, last_obs, env_state)
    combined_q_vals = conditional_q_vals * learned_q_vals
    return combined_q_vals