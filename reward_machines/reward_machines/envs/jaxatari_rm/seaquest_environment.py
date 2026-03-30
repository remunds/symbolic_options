import gym
from gym import spaces
import numpy as np
import os

import jax
import jaxatari
from jaxatari.wrappers import AtariWrapper, FlattenObservationWrapper, ObjectCentricWrapper

from reward_machines.rm_environment import RewardMachineEnv
from envs.jaxatari_rm.seaquest_logging import (
    total_collected,
    total_rescued,
    total_shot,
    total_surface_without_dying,
)


class JaxSeaquestEnv(gym.Env):
    """Gym-compatible adapter for the JAXAtari Seaquest environment."""

    def __init__(self, seed=0):
        base_env = jaxatari.make("seaquest")
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
        prev_state, current_state: SeaquestObservation instances.
        """
        SURFACE_Y = 46
        OXYGEN_CRITICAL = 15

        prev_env = self._env_state(prev_state)
        curr_env = self._env_state(current_state)
        
        events = ''
        # Event: 'd' (Died / Terminal)
        if self._to_scalar(curr_env.lives) <= 0 and self._to_scalar(prev_env.lives) > 0:
            events += 'd'
        
        # Event: 's' (Surfaced successfully)
        if self._to_float(curr_env.player_y) <= SURFACE_Y and self._to_float(prev_env.player_y) > SURFACE_Y:
            events += 's'
            
        # Event: 'f' (Full capacity or low oxygen - need to surface)
        is_full = self._to_scalar(curr_env.divers_collected) == 6 and self._to_scalar(prev_env.divers_collected) < 6
        is_choking = self._to_scalar(curr_env.oxygen) <= OXYGEN_CRITICAL and self._to_scalar(prev_env.oxygen) > OXYGEN_CRITICAL
        if is_full or is_choking:
            events += 'f'
            
        # Default: no notable state-changing event occurred
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
        rescued_step = self._to_float(total_rescued(prev_env, next_env))
        collected_step = self._to_float(total_collected(prev_env, next_env))
        shot_step = self._to_float(total_shot(prev_env, next_env))
        surface_without_dying_step = self._to_float(total_surface_without_dying(prev_env, next_env))

        self._last_events = self._compute_events(self._prev_state, next_state, reward)
        self._state = next_state
        self._prev_state = next_state

        obs = np.asarray(obs, dtype=np.float32)
        done = bool(np.asarray(done))

        info = dict(info)
        metrics = {
            "seaquest_total_rescued": rescued_step,
            "seaquest_total_collected": collected_step,
            "seaquest_total_shot": shot_step,
            "seaquest_total_surface_without_dying": surface_without_dying_step,
        }
        info.update({
            "original_reward": env_reward,
            "env_reward": env_reward,
            "metrics": metrics,
            "seaquest_total_rescued": rescued_step,
            "seaquest_total_collected": collected_step,
            "seaquest_total_shot": shot_step,
            "seaquest_total_surface_without_dying": surface_without_dying_step,
            "prev_lives": self._to_scalar(prev_env.lives),
            "lives": self._to_scalar(next_env.lives),
            "prev_collected_divers": self._to_scalar(prev_env.divers_collected),
            "collected_divers": self._to_scalar(next_env.divers_collected),
            "prev_player_score": self._to_scalar(prev_env.score),
            "player_score": self._to_scalar(next_env.score),
            "prev_player_y": self._to_float(prev_env.player_y),
            "player_y": self._to_float(next_env.player_y),
            "prev_oxygen_level": self._to_scalar(prev_env.oxygen),
            "oxygen_level": self._to_scalar(next_env.oxygen),
        })

        return obs, env_reward, done, info

    def reset(self):
        self._rng, reset_key = jax.random.split(self._rng)
        obs, self._state = self.env.reset(reset_key)

        self._prev_state = self._state
        self._last_events = ""

        return np.asarray(obs, dtype=np.float32)


class JaxSeaquestRMEnv1(RewardMachineEnv):
    def __init__(self):
        env = JaxSeaquestEnv()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rm_files = [os.path.join(base_dir, "reward_machines", "t1_sea.txt")]
        super().__init__(env, rm_files)


class JaxSeaquestRMEnv2(RewardMachineEnv):
    def __init__(self):
        env = JaxSeaquestEnv()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        rm_files = [os.path.join(base_dir, "reward_machines", "t2_sea.txt")]
        super().__init__(env, rm_files)
