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

# Multiple things:
# 1. the q_values need to be normalized (s.t. they are comparable)
#    -> use rewards between 0 and 1
# 2. We may want to use reward-shaping to ensure that they rewards still try to solve the overall env goal

# The following somehow worked -> but tried to stay playing as long as possible to accumulate most reward
# even with combined mode, it still only reaches ±18 

# def track_and_align(prev_state, state):
#     state = unpack(state)
#     prev_state = unpack(prev_state)
#     # env_reward = JaxPong()._get_env_reward(prev_state, state)
#     align_penalty = -jnp.abs((state.player_y+10) - state.ball_y)
#     # align_penalty_prev = -jnp.abs((prev_state.player_y+10) - prev_state.ball_y)
#     # return env_reward + GAMMA * align_penalty_prev - align_penalty

#     # rewrite for positive reward
#     # error_prev = jnp.abs((prev_state.player_y + 10) - prev_state.ball_y) 
#     # error = jnp.abs((state.player_y + 10) - state.ball_y)
#     # align_reward = jnp.where(error < error_prev, 1.0, -1.0)

#     return align_penalty

# def return_shot(prev_state, state):
#     state = unpack(state)
#     prev_state = unpack(prev_state)
#     # env_reward = JaxPong()._get_env_reward(prev_state, state)

#     def hit_reward(prev_state, state):
#         prev_move_right = prev_state.ball_vel_x > 0
#         move_left = state.ball_vel_x < 0
#         # x_change = jnp.sign(prev_state.ball_vel_x) != jnp.sign(state.ball_vel_x)
#         close_to_player = state.ball_x >= (PLAYER_X - 5)
#         # If the ball is close to the player and the direction has changed, we consider it a hit
#         hit = jnp.logical_and(move_left, close_to_player)
#         hit = jnp.logical_and(hit, prev_move_right)
#         return jnp.where(hit, 1.0 + 0.1 * jnp.abs(state.ball_vel_x), 0.0) 

#     # return lax.cond(hit, hit_reward, lambda _: 0.0, operand=None)
#     # return env_reward + GAMMA * hit_reward(prev_state) - hit_reward(state)
#     return hit_reward(prev_state, state)

# def defensive_positioning(prev_state, state):
#     state = unpack(state)
#     # prev_state = unpack(prev_state)
#     # env_reward = JaxPong()._get_env_reward(prev_state, state)
#     # point_lost = state.enemy_score > prev_state.enemy_score

#     def center_bonus_fn(state):
#         speed = jnp.sqrt(state.ball_vel_x ** 2 + state.ball_vel_y ** 2)
#         return jnp.where(speed > 0.02, 0.5 / (jnp.abs((state.player_y+10) - BALL_START_Y) + 1e-3), 0.0) 

#     # center_bonus = lax.cond(speed > 0.02, center_bonus_fn, lambda _: 0.0, operand=None)
#     # loss_penalty = lax.cond(point_lost, lambda _: -2.0, lambda _: 0.0, operand=None)

#     return center_bonus_fn(state)
#     # return env_reward + GAMMA * center_bonus_fn(prev_state) - center_bonus_fn(state)

# class Skill(IntEnum):
#     TRACK_ALIGN = 0
#     RETURN_SHOT = 1
#     DEFENSIVE_POS = 2

# def llm_meta_policy(network, meta_train_state, last_obs, state):
#     state = unpack(state)
#     player_ball_y_dist = jnp.abs((state.player_y + 10) - state.ball_y)
#     # aligned_y = player_ball_y_dist <= 10
#     near_player_x = state.ball_x >= (PLAYER_X - 70) 
#     incoming = state.ball_vel_x > 0

#     #1: aligned_y and ball_close -> return shot
#     #2: incoming -> track
#     #3: else -> defensive position


#     # 1. Return shot if aligned & ball near
#     # return_shot_cond = jnp.logical_and(aligned_y, near_player_x)
#     return_shot_cond = near_player_x 

#     q_vals = jnp.where(
#         return_shot_cond,
#         Skill.RETURN_SHOT,
#         jnp.where(incoming, Skill.TRACK_ALIGN, Skill.DEFENSIVE_POS)
#     )
#     q_vals = jax.nn.one_hot(q_vals, len(Skill))

#     return q_vals

# def conditional_meta_policy(network, meta_train_state, last_obs, state):
#     # always active: defensive positioning
#     state = unpack(state)
#     player_ball_y_dist = jnp.abs((state.player_y + 10) - state.ball_y)
#     # aligned_y = player_ball_y_dist <= 10
#     near_player_x = state.ball_x >= (PLAYER_X - 70) 
#     incoming = state.ball_vel_x > 0

#     #1: aligned_y and ball_close -> return shot
#     #2: incoming -> track
#     #3: else -> defensive position

#     # 1. Return shot if aligned & ball near
#     # return_shot_cond = jnp.logical_and(aligned_y, near_player_x)
#     return_shot_cond = near_player_x
#     return_shot_qvals = jax.nn.one_hot(
#         jnp.where(
#             return_shot_cond, Skill.RETURN_SHOT, Skill.DEFENSIVE_POS
#         ),
#         len(Skill)
#     )

#     # 2. Track and align if incoming
#     track_align_qvals = jax.nn.one_hot(
#         jnp.where(
#             incoming, Skill.TRACK_ALIGN, Skill.DEFENSIVE_POS
#         ),
#         len(Skill)
#     )

#     # 3. Defensive positioning
#     defensive_pos_qvals = jax.nn.one_hot(
#         jnp.where(
#             jnp.ones_like(incoming), Skill.DEFENSIVE_POS, Skill.DEFENSIVE_POS
#         ),
#         len(Skill)
#     )

#     # Combine the Q-values
#     q_vals = jnp.logical_or(
#         jnp.logical_or(return_shot_qvals, track_align_qvals),
#         defensive_pos_qvals
#     )

#     return q_vals

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
    # prev_dist = abs((prev.player_y+PLAYER_H/2) - prev.ball_y)
    # curr_dist = abs((curr.player_y+PLAYER_H/2) - curr.ball_y)
    # reward = jnp.where(
    #     curr_dist < prev_dist,  # distance reduced
    #     1.0, 
    #     jnp.where(curr_dist > prev_dist, -1.0, 0.0)  # distance increased or unchanged
    # )
    # delta = prev_dist - curr_dist
    # reward = jnp.maximum(0.0, jnp.minimum(1.0, delta / 0.2))  # scale with small range

    distance = abs((curr.player_y + PLAYER_H / 2) - curr.ball_y)
    reward = jnp.where(distance < 3, 1.0, 0.0)
    return env_reward + reward * 0.1 # scale down 
    # return reward 
    # return reward

def return_shot(prev: PongState, curr: PongState) -> float:
    """
    Binary reward ∈ {0, 1} if paddle and ball align closely at contact range.
    """
    env_reward = JaxPong()._get_reward(prev, curr)
    x_close = abs(curr.ball_x - PLAYER_X) < 1.5 * PLAYER_W 
    y_align = abs(curr.ball_y - curr.player_y) < 0.5 * PLAYER_H
    ball_speed = abs(curr.ball_x - prev.ball_x) * 0.1
    reward = jnp.where(x_close & y_align, 1.0 + ball_speed, 0.0)

    return env_reward + reward * 0.1
    # return reward

def defensive_positioning(prev: PongState, curr: PongState) -> float:
    """
    Reward ∈ [0, 1] based on movement toward screen vertical center.
    1 for maximal improvement, 0 if worsened.
    """
    screen_center_y = SCREEN_CENTER_Y 
    env_reward = JaxPong()._get_reward(prev, curr)
    # distance to mid
    dist = abs((curr.player_y + PLAYER_H / 2) - screen_center_y)
    reward = jnp.where(dist < 3, 1.0, 0.0)
    # return reward
    # prev_dist = abs(prev.player_y - screen_center_y)
    # curr_dist = abs(curr.player_y - screen_center_y)
    # delta = prev_dist - curr_dist
    # reward = jnp.maximum(0.0, jnp.minimum(1.0, delta / 0.2))  # scale with small range
    return env_reward + reward * 0.1  # scale down 
    # return reward

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

    # if ball_approaching and close_to_paddle and y_misaligned:
    #     return 1  # InterceptBall
    # elif ball_approaching:
    #     return 0  # TrackBall
    # elif not near_center:
    #     return 2  # RecoverToCenter
    # else:
    #     return 0  # default to tracking
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