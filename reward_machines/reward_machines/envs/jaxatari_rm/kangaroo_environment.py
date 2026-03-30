import gym
from gym import spaces
import numpy as np
import os

import jax
import jax.numpy as jnp
import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper
from jaxatari.games.jax_kangaroo import JaxKangaroo

from reward_machines.rm_environment import RewardMachineEnv
from envs.jaxatari_rm.kangaroo_logging import (
    enemies_killed,
    fruits_collected,
    reached_platform_level,
)


class JaxKangarooEnv(gym.Env):
    """Gym-compatible adapter for the JAXAtari Kangaroo environment."""

    def __init__(self, seed=0):
        base_env = jaxatari.make("kangaroo")
        self._game = JaxKangaroo()
        self.env = FlattenObservationWrapper(ObjectCentricWrapper(AtariWrapper(base_env)))

        raw_action_space = self.env.action_space()
        self.action_space = spaces.Discrete(int(raw_action_space.n))

        self._rng = None
        self._state = None
        self._prev_state = None
        self._last_events = ""
        self._seed_value = 0

        self.seed(seed)
        initial_obs = self.reset()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial_obs.shape,
            dtype=np.float32,
        )

    def seed(self, seed=None):
        if seed is None:
            seed = 0
        self._seed_value = int(seed)
        self._rng = jax.random.PRNGKey(self._seed_value)
        return [self._seed_value]

    def _to_scalar(self, value):
        return int(np.asarray(value).item())

    def _to_float(self, value):
        return float(np.asarray(value).item())

    def _env_state(self, state):
        return state.env_state if hasattr(state, "env_state") else state

    def _compute_events(self, prev_state, current_state, reward):
        """
        Evaluates the observations to return an event string.
        prev_state, current_state: KangarooObservation instances.
        """
        prev_env = self._env_state(prev_state)
        curr_env = self._env_state(current_state)
        prev_obs = self._game._get_observation(prev_env)
        curr_obs = self._game._get_observation(curr_env)

        events = ''
        # Check if child is reached
        dist_to_child = jnp.sqrt((curr_obs.player.x - curr_obs.child.x)**2 + 
                                (curr_obs.player.y - curr_obs.child.y)**2)
        if dist_to_child < 10:
            events += 'g'

        # Check for nearby hazards (monkeys or thrown apples)
        # We check if any active hazard is within a 'danger zone'
        hazard_x = jnp.concatenate([curr_obs.monkeys.x, curr_obs.thrown_coconuts.x])
        hazard_active = jnp.concatenate([curr_obs.monkeys.active, curr_obs.thrown_coconuts.active])
        
        distances = jnp.abs(hazard_x - curr_obs.player.x)
        # Filter only active hazards
        nearby_hazards = jnp.any((distances < 30) & (hazard_active == 1))

        if nearby_hazards:
            events += 'h'
        else:
            events += 'a'
        return events

    def get_events(self):
        return self._last_events

    def step(self, action):
        action = np.int32(action)
        prev_state = self._state
        prev_env = self._env_state(prev_state)
        obs, next_state, reward, done, info = self.env.step(self._state, action)
        env_reward = self._to_float(reward)
        next_env = self._env_state(next_state)
        prev_obs = self._game._get_observation(prev_env)
        next_obs = self._game._get_observation(next_env)

        prev_child_x = self._to_float(prev_env.level.child_position[0])
        prev_child_y = self._to_float(prev_env.level.child_position[1])
        child_x = self._to_float(next_env.level.child_position[0])
        child_y = self._to_float(next_env.level.child_position[1])
        prev_active_monkeys = int(np.count_nonzero(np.asarray(prev_env.level.monkey_states)))
        active_monkeys = int(np.count_nonzero(np.asarray(next_env.level.monkey_states)))
        prev_active_fruits = int(np.count_nonzero(np.asarray(prev_env.level.fruit_actives)))
        active_fruits = int(np.count_nonzero(np.asarray(next_env.level.fruit_actives)))
        platform_step = self._to_float(reached_platform_level(prev_env, next_env))
        enemy_killed_step = self._to_float(enemies_killed(prev_env, next_env))
        fruit_collected_step = self._to_float(fruits_collected(prev_env, next_env))

        self._last_events = self._compute_events(self._prev_state, next_state, reward)
        self._state = next_state
        self._prev_state = next_state

        obs = np.asarray(obs, dtype=np.float32)
        done = bool(np.asarray(done))

        info = dict(info)
        metrics = {
            "kangaroo_reached_platform_level": platform_step,
            "kangaroo_enemies_killed": enemy_killed_step,
            "kangaroo_fruits_collected": fruit_collected_step,
        }
        info.update({
            "original_reward": env_reward,
            "env_reward": env_reward,
            "metrics": metrics,
            "kangaroo_reached_platform_level": platform_step,
            "kangaroo_enemies_killed": enemy_killed_step,
            "kangaroo_fruits_collected": fruit_collected_step,
            "prev_lives": self._to_scalar(prev_env.lives),
            "lives": self._to_scalar(next_env.lives),
            "prev_player_score": self._to_scalar(prev_env.score),
            "player_score": self._to_scalar(next_env.score),
            "prev_current_level": self._to_scalar(prev_env.current_level),
            "current_level": self._to_scalar(next_env.current_level),
            "prev_player_x": self._to_float(prev_env.player.x),
            "player_x": self._to_float(next_env.player.x),
            "prev_player_y": self._to_float(prev_env.player.y),
            "player_y": self._to_float(next_env.player.y),
            "prev_child_x": prev_child_x,
            "child_x": child_x,
            "prev_child_y": prev_child_y,
            "child_y": child_y,
            "prev_active_monkeys": prev_active_monkeys,
            "active_monkeys": active_monkeys,
            "prev_active_fruits": prev_active_fruits,
            "active_fruits": active_fruits,
            "prev_player_active": self._to_scalar(prev_obs.player.active),
            "player_active": self._to_scalar(next_obs.player.active),
        })

        return obs, env_reward, done, info

    def reset(self):
        self._rng, reset_key = jax.random.split(self._rng)
        obs, self._state = self.env.reset(reset_key)

        self._prev_state = self._state
        self._last_events = ""

        return np.asarray(obs, dtype=np.float32)


class JaxKangarooRMEnv1(RewardMachineEnv):
    def __init__(self):
        env = JaxKangarooEnv()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rm_files = [os.path.join(base_dir, "reward_machines", "t1_kang.txt")]
        super().__init__(env, rm_files)


class JaxKangarooRMEnv2(RewardMachineEnv):
    def __init__(self):
        env = JaxKangarooEnv()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rm_files = [os.path.join(base_dir, "reward_machines", "t2_kang.txt")]
        super().__init__(env, rm_files)
