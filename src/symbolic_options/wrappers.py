"""Wrappers for pure RL."""

import functools
from typing import Any, Dict, Tuple, Union


import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from jaxtari.environment import EnvState
from gymnax.environments import spaces


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    #   def __init__(self, env: env.env):
    #     super().__init__(env)

    def observation_space(self) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space().low,
            high=self._env.observation_space().high,
            shape=(np.prod(self._env.observation_space().shape),),
            dtype=self._env.observation_space().dtype,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, EnvState]:
        obs, state = self._env.reset(key)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, EnvState, float, bool, Any]:  # dict]:
        obs, state, reward, done, info = self._env.step(key, state, action)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    #   def __init__(self, env: env.env):
    #     super().__init__(env)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(env_state, 0, 0, 0, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        
    ) -> Tuple[chex.Array, LogEnvState, jnp.ndarray, bool, Dict[Any, Any]]:
        """Step the env.


        Args:
          key: PRNG key.
          state: The current state of the env.
          action: The action to take.


        Returns:
          A tuple of (observation, state, reward, done, info).
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info

@struct.dataclass
class MultiRewardLogEnvState:
    env_state: EnvState
    episode_returns_env: float
    episode_returns: chex.Array#[float]
    episode_lengths: int
    returned_episode_returns_env: float
    returned_episode_returns: chex.Array#[float]
    returned_episode_lengths: int

class MultiRewardLogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    #   def __init__(self, env: env.env):
    #     super().__init__(env)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, 
    ) -> Tuple[chex.Array, MultiRewardLogEnvState]:
        obs, env_state = self._env.reset(key)
        dummy_info = self._env.step(key, env_state, 0)[4]
        episode_returns_init = jnp.zeros_like(dummy_info["all_rewards"])
        state = MultiRewardLogEnvState(env_state, 0, episode_returns_init, 0, 0, episode_returns_init, 0)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MultiRewardLogEnvState,
        action: Union[int, float],
        
    ) -> Tuple[chex.Array, MultiRewardLogEnvState, jnp.ndarray, bool, Dict[Any, Any]]:
        """Step the env.
        Args:
          key: PRNG key.
          state: The current state of the env.
          action: The action to take.

        Returns:
          A tuple of (observation, state, reward, done, info).
        """
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        new_episode_return_env = state.episode_returns_env + reward 
        new_episode_return = state.episode_returns + info["all_rewards"]
        new_episode_length = state.episode_lengths + 1
        state = MultiRewardLogEnvState(
            env_state=env_state,
            episode_returns_env=new_episode_return_env * (1 - done),
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns_env=state.returned_episode_returns_env * (1 - done)
            + new_episode_return_env * done,
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
        )
        info["returned_episode_env_returns"] = state.returned_episode_returns_env
        for i, r in enumerate(new_episode_return):
            info[f"returned_episode_returns_{i}"] = state.returned_episode_returns[i]
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = done
        return obs, state, reward, done, info