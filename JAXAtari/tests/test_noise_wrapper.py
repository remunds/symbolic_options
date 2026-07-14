"""Tests for NoiseWrapper from wrappers_new.py."""

import collections

import jax
import jax.numpy as jnp
import pytest
import jaxatari.spaces as spaces
import numpy as np
from jaxatari.wrappers_new import (
    AtariWrapper,
    NoiseWrapper,
    ObjectCentricWrapper,
)


class FakeObsEnv:
    """A minimal fake environment with pytree observations for testing NoiseWrapper."""

    Obs = collections.namedtuple("Obs", ["player", "ball", "enemy"])

    def __init__(self):
        self._observation_space = spaces.Dict({
            "player": spaces.Box(low=0, high=255, shape=(2,), dtype=jnp.uint8),
            "ball": spaces.Box(low=0, high=255, shape=(2,), dtype=jnp.uint8),
            "enemy": spaces.Box(low=0, high=255, shape=(4,), dtype=jnp.uint8),
        })
        self.ACTION_SET = jnp.arange(6, dtype=jnp.int32)

    def observation_space(self):
        return self._observation_space

    def image_space(self):
        return spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    def reset(self, key):
        return self.Obs(
            player=jnp.array([100, 50], dtype=jnp.uint8),
            ball=jnp.array([80, 120], dtype=jnp.uint8),
            enemy=jnp.array([30, 40, 50, 60], dtype=jnp.uint8),
        ), jnp.array(0, dtype=jnp.int32)

    def step(self, state, action):
        state = state + 1
        return self.Obs(
            player=jnp.array([100 + state, 50], dtype=jnp.uint8),
            ball=jnp.array([80, 120], dtype=jnp.uint8),
            enemy=jnp.array([30, 40, 50, 60], dtype=jnp.uint8),
        ), state, 1.0, False, {}

    def render(self, state):
        return jnp.zeros((210, 160, 3), dtype=jnp.uint8)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_noise_wrapper_noop_passthrough():
    """With defaults (detect_prob=1.0, std_dev=0.0), NoiseWrapper is a no-op."""
    atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    noise_env = NoiseWrapper(atari_env, detect_prob=1.0, std_dev=0.0)

    key = jax.random.PRNGKey(0)
    obs_noisy, state_noisy = noise_env.reset(key)

    # Compare with unwrapped AtariWrapper
    atari_env2 = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    obs_clean, state_clean = atari_env2.reset(key)

    assert jnp.array_equal(obs_noisy.player, obs_clean.player)
    assert jnp.array_equal(obs_noisy.ball, obs_clean.ball)
    assert jnp.array_equal(obs_noisy.enemy, obs_clean.enemy)

    # Step once
    obs_noisy, state_noisy, *_ = noise_env.step(state_noisy, 0)
    obs_clean, state_clean, *_ = atari_env2.step(state_clean, 0)

    assert jnp.array_equal(obs_noisy.player, obs_clean.player)
    assert jnp.array_equal(obs_noisy.ball, obs_clean.ball)


def test_noise_wrapper_space_passthrough():
    """NoiseWrapper passes through the observation space unchanged."""
    atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    noise_env = NoiseWrapper(atari_env, detect_prob=0.8, std_dev=1.5)

    base_space = atari_env.observation_space()
    noise_space = noise_env.observation_space()

    assert isinstance(noise_space, spaces.Dict)
    assert noise_space.spaces.keys() == base_space.spaces.keys()

    for key in base_space.spaces:
        assert noise_space.spaces[key].shape == base_space.spaces[key].shape
        assert noise_space.spaces[key].dtype == base_space.spaces[key].dtype


def test_noise_wrapper_gaussian_noise_changes_obs():
    """With std_dev > 0, observations should differ from the original."""
    atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    noise_env = NoiseWrapper(atari_env, detect_prob=1.0, std_dev=50.0)

    key = jax.random.PRNGKey(42)
    obs_noisy, state = noise_env.reset(key)

    atari_env2 = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    obs_clean, _ = atari_env2.reset(key)

    assert not jnp.array_equal(obs_noisy.player, obs_clean.player)
    assert not jnp.array_equal(obs_noisy.ball, obs_clean.ball)
    assert obs_noisy.player.shape == obs_clean.player.shape
    assert obs_noisy.player.dtype == obs_clean.player.dtype


def test_noise_wrapper_dropping_zeros_out_leaves():
    """With detect_prob=0.0, ALL leaves should be zeroed out."""
    atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    noise_env = NoiseWrapper(atari_env, detect_prob=0.0, std_dev=0.0)

    key = jax.random.PRNGKey(7)
    obs, state = noise_env.reset(key)

    assert jnp.all(obs.player == 0)
    assert jnp.all(obs.ball == 0)
    assert jnp.all(obs.enemy == 0)

    obs, state, *_ = noise_env.step(state, 0)
    assert jnp.all(obs.player == 0)
    assert jnp.all(obs.ball == 0)


def test_noise_wrapper_deterministic_with_same_seed():
    """Same seed should produce identical noisy observations."""
    def make_noisy(seed):
        atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
        noise_env = NoiseWrapper(atari_env, detect_prob=0.8, std_dev=10.0)
        return noise_env.reset(jax.random.PRNGKey(seed))[0]

    obs1 = make_noisy(123)
    obs2 = make_noisy(123)

    assert jnp.array_equal(obs1.player, obs2.player)
    assert jnp.array_equal(obs1.ball, obs2.ball)


def test_noise_wrapper_different_seed_different_noise():
    """Different seeds should produce different noise patterns."""
    def make_noisy(seed):
        atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
        noise_env = NoiseWrapper(atari_env, detect_prob=0.5, std_dev=50.0)
        return noise_env.reset(jax.random.PRNGKey(seed))[0]

    obs1 = make_noisy(1)
    obs2 = make_noisy(2)

    differs = (
        not jnp.array_equal(obs1.player, obs2.player)
        or not jnp.array_equal(obs1.ball, obs2.ball)
    )
    assert differs, "Different seeds should produce different noise"


def test_noise_wrapper_partial_dropping():
    """With detect_prob=0.5, roughly half the leaves should be zeroed."""
    zero_count = 0
    for seed in range(100):
        atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
        noise_env = NoiseWrapper(atari_env, detect_prob=0.5, std_dev=0.0)
        obs, _ = noise_env.reset(jax.random.PRNGKey(seed))
        if jnp.all(obs.player == 0):
            zero_count += 1
        if jnp.all(obs.ball == 0):
            zero_count += 1
        if jnp.all(obs.enemy == 0):
            zero_count += 1

    # 300 total leaves (100 seeds × 3), expect ~150 zeros
    assert 50 < zero_count < 250, f"Expected ~150 zeros, got {zero_count}/300"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_noise_wrapper_integration_with_object_centric():
    """NoiseWrapper → ObjectCentricWrapper chain works end-to-end."""
    atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    noise_env = NoiseWrapper(atari_env, detect_prob=0.9, std_dev=5.0)
    env = ObjectCentricWrapper(noise_env, frame_stack_size=4, frame_skip=1)

    key = jax.random.PRNGKey(99)
    obs, state = env.reset(key)

    assert obs.ndim == 2
    assert obs.shape[0] == 4
    assert obs.dtype == jnp.float32

    for _ in range(5):
        obs, state, reward, terminated, truncated, info = env.step(state, 0)
        assert obs.shape[0] == 4
        assert obs.dtype == jnp.float32


def test_noise_wrapper_multiple_steps():
    """NoiseWrapper works correctly across many steps."""
    atari_env = AtariWrapper(FakeObsEnv(), first_fire=False, episodic_life=False)
    noise_env = NoiseWrapper(atari_env, detect_prob=0.8, std_dev=3.0)

    key = jax.random.PRNGKey(55)
    obs, state = noise_env.reset(key)

    for _ in range(50):
        obs, state, reward, terminated, truncated, info = noise_env.step(state, 0)
        assert obs.player.shape == (2,)
        assert obs.ball.shape == (2,)
        assert obs.enemy.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__])
