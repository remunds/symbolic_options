import jax.numpy as jnp
from jax import lax
import jax
from enum import IntEnum
from jaxatari.games.jax_freeway import FreewayState, FreewayConstants, JaxFreeway

def unpack(state):
    while not isinstance(state, FreewayState):
        if hasattr(state, 'atari_state'):
            state = state.atari_state
        elif hasattr(state, 'env_state'):
            state = state.env_state
        else:
            raise ValueError("State is not a FreewayState or does not contain a FreewayState.")
    return state

def env_reward(prev_state: FreewayState, state: FreewayState) -> float:
    """
    Compute the environment reward based on the previous and current state.
    """
    prev_state = unpack(prev_state)
    state = unpack(state)
    # Compute the environment reward based on the previous and current state
    return JaxFreeway()._get_reward(prev_state, state)

def avoid_crash(prev: FreewayState, curr: FreewayState) -> float:
    # Check for collisions
    def check_collision(car_pos):
        car_x, car_y = car_pos
        return jnp.logical_and(
            FreewayConstants().chicken_x < car_x + FreewayConstants().car_width,
            jnp.logical_and(
                FreewayConstants().chicken_x + FreewayConstants().chicken_width > car_x,
                jnp.logical_and(
                    curr.chicken_y - FreewayConstants().chicken_height < car_y,
                    curr.chicken_y > car_y - FreewayConstants().car_height,
                ),
            ),
        )
    # vmap over all cars
    collisions = jax.vmap(check_collision, in_axes=(0,))(curr.cars)
    # Check if any collision occurred
    collision_occurred = jnp.any(collisions, axis=-1)
    # movement = curr.chicken_y < prev.chicken_y
    return jnp.where(collision_occurred, -1.0, 0.0)

    # # maximize distance to closest car 
    # # cars: [N, 2] (x, y) coordinates of cars
    # distance_y = jnp.abs(curr.chicken_y - curr.cars[..., 1])
    # distance_x = jnp.abs(FreewayConstants().chicken_x - curr.cars[..., 0])
    # total_distance = jnp.sqrt(distance_x**2 + distance_y**2)
    # min_distance = jnp.min(total_distance, axis=-1) # over all cars

    # # prev_distance_y = jnp.abs(prev.chicken_y - prev.cars[..., 1])
    # # prev_distance_x = jnp.abs(FreewayConstants().chicken_x - prev.cars[..., 0])
    # # prev_total_distance = jnp.sqrt(prev_distance_x**2 + prev_distance_y**2)
    # # prev_min_distance = jnp.min(prev_total_distance, axis=-1) # over all cars

    # # reward = jnp.where(
    # #     min_distance > prev_min_distance,
    # #     1.0, # moved away from car
    # #     0.0
    # # )
    # # env_reward = curr.score - prev.score
    # # return env_reward + reward/10

    # # chatgpt reward: 1-exp(-d/theta)
    # # reward = 1-jnp.exp(-min_distance / 0.1)
    # reward = 1-jnp.exp(-min_distance/1000)
    # reward = jnp.where(collision_occurred, -10.0, reward)  # penalize collisions
    # return reward

def go_forward(prev: FreewayState, curr: FreewayState) -> float:
    # reward for moving forward
    # chicken_y is the vertical position of the chicken
    # if chicken_y is greater than previous, it moved forward
    reward = jnp.where(curr.chicken_y < prev.chicken_y, 1.0, 0.0)
    # env_reward = curr.score - prev.score
    # return env_reward + reward / 100
    return reward


def llm_meta_policy(network, meta_train_state, last_obs, env_state):
    """
    Priority: avoid_crash > go_forward
    """
    state = unpack(env_state)

    #activate avoid_crash based on current and upcoming lane, not total distance

    distance_y = jnp.abs(jnp.expand_dims(state.chicken_y, axis=1) - state.cars[..., 1])
    cars_in_front = state.cars[..., 1] < (jnp.expand_dims(state.chicken_y, axis=1) + FreewayConstants().lane_spacing)
    cars_in_close_front = jnp.logical_and(cars_in_front, distance_y < FreewayConstants().lane_spacing * 2)
    distance_x = jnp.abs(FreewayConstants().chicken_x - state.cars[..., 0])
    total_distance = jnp.sqrt(distance_x**2 + distance_y**2)
    x_distance_filtered = jnp.where(
        cars_in_close_front,
        distance_x,
        jnp.inf  # set to inf if car is not in close front
    )
    min_x_distance = jnp.min(x_distance_filtered, axis=-1)  # over all cars

    # car_close = min_x_distance < FreewayConstants().chicken_width * 4
    car_close = (total_distance < FreewayConstants().chicken_width * 3).any(axis=-1)  # check if any car is close in any lane

    qvals = jnp.where(
        car_close,
        0,  # avoid_crash
        1
    )

    qvals = jax.nn.one_hot(qvals, 2) 
    return qvals.astype(jnp.float32)  # Avoid crash or go forward 

def conditional_meta_policy(network, meta_train_state, last_obs, env_state):
    # default to going_forward (always active) 
    # Same as llm_meta_policy, but allow multiple skills to be active at once 
    state = unpack(env_state)

    car_close = jnp.abs(state.cars[:, 1] - state.chicken_y) < 2 * FreewayConstants().chicken_width

    qvals_avoid_crash = jnp.where(
        car_close,
        0,
        1
    )
    qvals_avoid_crash = jax.nn.one_hot(qvals_avoid_crash, 2)  # Avoid crash or not
    qvals_go_forward = jnp.ones_like(qvals_avoid_crash)  # always go forward
    qvals_go_forward = jax.nn.one_hot(qvals_go_forward, 2)  # Go forward or not

    # Combine the Q-values
    q_vals = jnp.logical_or(
        qvals_avoid_crash, 
        qvals_go_forward
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