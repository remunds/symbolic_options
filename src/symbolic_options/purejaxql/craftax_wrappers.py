import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Dict, Optional, Tuple, Union, Any


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_util.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info
    
class NoNecessitiesWrapper(GymnaxWrapper):
    """
    Removes food, water, energy necessities from the environment. 
    -> Always keeps them at the maximum value.
    """
    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, state = self._env.reset(key, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        # replace state's necessities with the initial values 
        _, reset_state = self.reset(key, params)
        modif_state = state.replace(
            player_food=reset_state.player_food,
            player_drink=reset_state.player_drink,
            player_energy=reset_state.player_energy,
        )

        obs, new_state, reward, done, info = self._env.step(
            key, modif_state, action, params
        )
        
        return obs, new_state, reward, done, info

@struct.dataclass
class ExplorationState:
    env_state: Any
    exploration_map: chex.Array

class ExplorationMapWrapper(GymnaxWrapper):
    """Wrapper that tracks the exploration map of the environment."""
    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, state = self._env.reset(key, params)
        exp_state = ExplorationState(
            env_state=state,
            exploration_map=jnp.zeros_like(state.map)
        )
        return obs, exp_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        # replace state's necessities with the initial values 
        _, reset_state = self.reset(key, params)

        obs, new_state, reward, done, info = self._env.step(
            key, reset_state.env_state, action, params
        )

        exp_state = ExplorationState(
            env_state=new_state,
            exploration_map=state.exploration_map.at[new_state.player_position].set(1)
        )
        
        return obs, exp_state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
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
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
    
class MultiRewardWrapper(GymnaxWrapper):
    def __init__(self, env, reward_funcs=[]):
        super().__init__(env)
        self.reward_funcs = tuple(reward_funcs)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        return self._env.reset(key, params)
    
    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        # get rewards
        rewards = []
        for f in self.reward_funcs:
            rewards.append(f(state, env_state))
        rewards = jnp.stack(rewards, axis=0)
        
        info["all_rewards"] = rewards

        return obs, env_state, reward, done, info
    
@struct.dataclass
class MultiRewardLogEnvState:
    env_state: Any
    episode_returns_env: float
    episode_returns: chex.Array#[float]
    episode_lengths: int
    returned_episode_returns_env: float
    returned_episode_returns: chex.Array#[float]
    returned_episode_lengths: int

class MultiRewardLogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,2))
    def reset(
        self, key: chex.PRNGKey, params=None
    ) -> Tuple[chex.Array, MultiRewardLogEnvState]:
        obs, env_state = self._env.reset(key, params)
        dummy_info = self._env.step(key, env_state, 0, params)[4]
        episode_returns_init = jnp.zeros_like(dummy_info["all_rewards"])
        state = MultiRewardLogEnvState(env_state, 0.0, episode_returns_init, 0, 0.0, episode_returns_init, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,4))
    def step(
        self,
        key: chex.PRNGKey,
        state: MultiRewardLogEnvState,
        action: Union[int, float],
        params=None,
        
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
            key, state.env_state, action, params
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