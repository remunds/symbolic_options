import os
import pickle
import random
from collections import deque
from collections import defaultdict
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
import optax
from baselines import logger
from flax import linen as nn

try:
    import wandb
except ImportError:
    wandb = None

from rtpt import RTPT


def _default_info_metrics_callback(context):
    info = context.get("info", {})
    metrics = info.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _collect_metric_callback_values(metric_callbacks, context):
    metric_totals = {}
    for callback in metric_callbacks:
        callback_metrics = callback(context)
        if callback_metrics is None:
            continue
        if not isinstance(callback_metrics, dict):
            raise TypeError("Metric callback must return dict[str, number] or None")
        for metric_name, metric_value in callback_metrics.items():
            if isinstance(metric_value, Number):
                key = str(metric_name)
                metric_totals[key] = metric_totals.get(key, 0.0) + float(metric_value)
    return metric_totals


def _unwrap_env(env):
    cur = env
    visited = set()
    while hasattr(cur, "env") and id(cur) not in visited:
        if hasattr(cur, "_state") and hasattr(cur, "_game"):
            return cur
        visited.add(id(cur))
        cur = cur.env
    return cur


def _get_env_name(env):
    base_env = _unwrap_env(env)
    spec = getattr(base_env, "spec", None)
    spec_id = getattr(spec, "id", None)
    if spec_id:
        return str(spec_id)
    return base_env.__class__.__name__.lower()


def _get_jaxatari_state(env):
    base_env = _unwrap_env(env)
    if hasattr(base_env, "_state"):
        state = base_env._state
        return state.env_state if hasattr(state, "env_state") else state
    return None


def _render_frame(env):
    state = _get_jaxatari_state(env)
    base_env = _unwrap_env(env)
    if state is not None and hasattr(base_env, "_game") and hasattr(base_env._game, "render"):
        return np.asarray(base_env._game.render(state), dtype=np.uint8)

    # Fallback for environments exposing Gym-style rgb_array rendering.
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()
    if frame is None:
        raise RuntimeError("Environment render() returned None during video capture")
    return np.asarray(frame, dtype=np.uint8)


def _capture_full_episode_video(env, q_net, params, current_obs, videos_dir, step_idx, max_steps=5000):
    try:
        import cv2
        os.makedirs(videos_dir, exist_ok=True)
        base_env = _unwrap_env(env)
        env_name = base_env.__class__.__name__.lower()
        video_path = os.path.join(videos_dir, "{}_step_{:08d}.mp4".format(env_name, step_idx))

        obs = env.reset()
        done = False
        writer = None
        steps = 0

        while not done and steps < int(max_steps):
            frame = _render_frame(env)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (w, h),
                )
            writer.write(frame[:, :, ::-1])

            q_values = q_net.apply(params, jnp.asarray(np.asarray(obs, dtype=np.float32)[None, :]))
            action = int(np.asarray(jnp.argmax(q_values, axis=1))[0])
            obs, _, done, _ = env.step(action)
            steps += 1

        if not done and steps >= int(max_steps):
            logger.log(
                "Stopped video capture at step limit ({} steps): {}".format(int(max_steps), video_path)
            )

        next_obs = env.reset()
        logger.log("Captured full-episode video ({} steps): {}".format(steps, video_path))
        return next_obs, video_path
    except Exception as ex:
        logger.log("Failed to capture full-episode video: {}".format(repr(ex)))
        try:
            return env.reset(), None
        except Exception:
            return current_obs, None
    finally:
        try:
            if "writer" in locals() and writer is not None:
                writer.release()
        except Exception:
            pass

class QNetwork(nn.Module):
    action_dim: int
    hidden_dims: tuple

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class ReplayBuffer(object):
    def __init__(self, obs_dim, capacity):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.pos
        self.obs[i] = np.asarray(obs, dtype=np.float32)
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = np.asarray(next_obs, dtype=np.float32)
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class CleanDQNModel(object):
    def __init__(self, q_net, params, hidden_dims):
        self.q_net = q_net
        self.params = params
        self.hidden_dims = hidden_dims
        self.initial_state = None

    def _act(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        q_values = self.q_net.apply(self.params, jnp.asarray(obs))
        actions = np.asarray(jnp.argmax(q_values, axis=1), dtype=np.int32)
        return int(actions[0]) if len(actions) == 1 else actions

    def step(self, observation, **kwargs):
        kwargs.pop("S", None)
        kwargs.pop("M", None)
        action = self._act(observation)
        return action, None, None, None

    def save(self, path):
        payload = {
            "params": jax.device_get(self.params),
            "hidden_dims": self.hidden_dims,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)


def learn(
    env,
    network="mlp",
    seed=None,
    use_crm=False,
    use_rs=False,
    lr=1e-4,
    total_timesteps=50_000_000, #50M * 4 = 200M frames
    buffer_size=50000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    train_freq=1,
    batch_size=32,
    print_freq=1000,
    learning_starts=1000,
    gamma=0.99,
    target_network_update_freq=500,
    num_layers=2,
    num_hidden=256,
    video_interval=100_000,
    video_length=600,
    **kwargs
):
    del network, video_length

    metric_callbacks = kwargs.pop("metric_callbacks", None)
    if metric_callbacks is None:
        metric_callbacks = []
    elif callable(metric_callbacks):
        metric_callbacks = [metric_callbacks]
    else:
        metric_callbacks = list(metric_callbacks)
    metric_callbacks = [_default_info_metrics_callback] + metric_callbacks

    use_wandb = True 
    wandb_project = "NEXUS_rebuttal2" 
    wandb_entity = "" 
    env_name = _get_env_name(env)
    wandb_name = "reward_machine_deepq_clean_{}".format(env_name)
    wandb_group = None 
    wandb_tags = None 
    wandb_mode = "online" 

    rtpt_name_initials = "RE" 
    rtpt_experiment_name = "reward_machine_deepq_clean_{}".format(seed) if seed is not None else "reward_machine_deepq_clean"

    rtpt = RTPT(
        name_initials=rtpt_name_initials,
        experiment_name=rtpt_experiment_name,
        max_iterations=int(total_timesteps),
    )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if use_crm:
        rm_states = env.get_num_rm_states()
        buffer_size = rm_states * buffer_size
        batch_size = rm_states * batch_size

    obs = env.reset()
    obs_dim = int(np.asarray(obs, dtype=np.float32).shape[0])
    action_dim = int(env.action_space.n)

    hidden_dims = tuple([int(num_hidden)] * int(num_layers))
    q_net = QNetwork(action_dim=action_dim, hidden_dims=hidden_dims)

    rng = jax.random.PRNGKey(0 if seed is None else int(seed))
    rng, init_key = jax.random.split(rng)
    params = q_net.init(init_key, jnp.zeros((1, obs_dim), dtype=jnp.float32))
    target_params = params

    optimizer = optax.adam(float(lr))
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, target_params, opt_state, obs_b, actions_b, rewards_b, next_obs_b, dones_b):
        def loss_fn(p):
            q_values = q_net.apply(p, obs_b)
            q_taken = jnp.take_along_axis(q_values, actions_b[:, None], axis=1).squeeze(axis=1)
            next_q = q_net.apply(target_params, next_obs_b)
            max_next_q = jnp.max(next_q, axis=1)
            targets = rewards_b + (1.0 - dones_b) * float(gamma) * max_next_q
            td_error = q_taken - jax.lax.stop_gradient(targets)
            return jnp.mean(jnp.square(td_error))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    replay = ReplayBuffer(obs_dim=obs_dim, capacity=buffer_size)
    np_rng = np.random.default_rng(seed)

    videos_dir = os.path.join(logger.get_dir() or ".", "videos")

    episode_rewards = [0.0]
    original_episode_rewards = [0.0]
    episode_length = 0
    env_returns = []
    original_env_returns = []
    env_lengths = []
    reward_total = 0.0
    original_reward_total = 0.0
    episode_custom_metrics = defaultdict(list)
    current_episode_custom_metrics = defaultdict(float)
    num_episodes = 0
    video_capture_max_steps = 5000
    last_loss = None
    wandb_run = None
    wandb_api = None

    if use_wandb:
        if wandb is None:
            raise ImportError("wandb logging requested but wandb is not installed. Install with `pip install wandb`.")
        wandb_api = wandb
        if isinstance(wandb_tags, str):
            wandb_tags = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
        wandb_config = {
            "algo": "deepq_clean",
            "seed": seed,
            "use_crm": use_crm,
            "use_rs": use_rs,
            "lr": lr,
            "total_timesteps": total_timesteps,
            "buffer_size": buffer_size,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "train_freq": train_freq,
            "batch_size": batch_size,
            "print_freq": print_freq,
            "learning_starts": learning_starts,
            "gamma": gamma,
            "target_network_update_freq": target_network_update_freq,
            "num_layers": num_layers,
            "num_hidden": num_hidden,
            "video_interval": video_interval,
        }
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            group=wandb_group,
            tags=wandb_tags,
            mode=wandb_mode,
            config=wandb_config,
            dir=logger.get_dir() or ".",
            reinit=True,
        )


    rtpt.start()
    for t in range(int(total_timesteps)):
        if video_interval and t > 0 and t % int(video_interval) == 0:
            obs, video_path = _capture_full_episode_video(
                env,
                q_net,
                params,
                obs,
                videos_dir,
                t,
                max_steps=video_capture_max_steps,
            )
            if wandb_run is not None and wandb_api is not None and video_path is not None:
                wandb_run.log({"video": wandb_api.Video(video_path, format="mp4")}, step=int(t))

        frac = min(float(t) / max(1.0, exploration_fraction * float(total_timesteps)), 1.0)
        epsilon = 1.0 + frac * (float(exploration_final_eps) - 1.0)

        if np_rng.random() < epsilon:
            action = int(np_rng.integers(action_dim))
        else:
            q_values = q_net.apply(params, jnp.asarray(np.asarray(obs, dtype=np.float32)[None, :]))
            action = int(np.asarray(jnp.argmax(q_values, axis=1))[0])

        new_obs, rew, done, info = env.step(action)
        original_rew = float(info.get("original_reward", info.get("env_reward", rew)))
        callback_context = {
            "t": t,
            "env": env,
            "obs": obs,
            "action": action,
            "rew": rew,
            "original_rew": original_rew,
            "new_obs": new_obs,
            "done": done,
            "info": info,
            "use_crm": use_crm,
            "use_rs": use_rs,
        }
        callback_metric_values = _collect_metric_callback_values(metric_callbacks, callback_context)
        for metric_name, metric_value in callback_metric_values.items():
            current_episode_custom_metrics[metric_name] += float(metric_value)

        if use_crm:
            experiences = info["crm-experience"]
        elif use_rs:
            experiences = [(obs, action, info["rs-reward"], new_obs, float(done))]
        else:
            experiences = [(obs, action, rew, new_obs, float(done))]

        for _obs, _action, _r, _new_obs, _done in experiences:
            replay.add(_obs, _action, _r, _new_obs, _done)

        obs = new_obs
        episode_rewards[-1] += rew
        original_episode_rewards[-1] += original_rew
        episode_length += 1
        reward_total += rew
        original_reward_total += original_rew

        if done:
            completed_return = float(episode_rewards[-1])
            completed_original_return = float(original_episode_rewards[-1])
            completed_length = int(episode_length)
            env_returns.append(completed_return)
            original_env_returns.append(completed_original_return)
            env_lengths.append(completed_length)
            for metric_name, metric_value in current_episode_custom_metrics.items():
                episode_custom_metrics[metric_name].append(float(metric_value))
            current_episode_custom_metrics.clear()
            obs = env.reset()
            episode_rewards.append(0.0)
            original_episode_rewards.append(0.0)
            episode_length = 0
            num_episodes += 1

        if t > learning_starts and t % int(train_freq) == 0 and replay.size >= int(batch_size):
            obs_b, actions_b, rewards_b, next_obs_b, dones_b = replay.sample(int(batch_size), np_rng)
            obs_b = jnp.asarray(obs_b)
            actions_b = jnp.asarray(actions_b)
            rewards_b = jnp.asarray(rewards_b)
            next_obs_b = jnp.asarray(next_obs_b)
            dones_b = jnp.asarray(dones_b)
            params, opt_state, loss = train_step(
                params, target_params, opt_state, obs_b, actions_b, rewards_b, next_obs_b, dones_b
            )
            last_loss = float(loss)

            if t % int(target_network_update_freq) == 0:
                target_params = params

        rtpt.step()

        if print_freq and t > 0 and t % int(print_freq) == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("total reward", reward_total)
            logger.record_tabular("original total reward", original_reward_total)
            logger.record_tabular("buffer_size", replay.size)
            logger.record_tabular("epsilon", epsilon)
            logger.record_tabular("loss", np.nan if last_loss is None else last_loss)

            episode_total_return = np.nan if len(env_returns) == 0 else float(env_returns[-1])
            episode_original_total_return = np.nan if len(original_env_returns) == 0 else float(original_env_returns[-1])
            episode_total_length = np.nan if len(env_lengths) == 0 else float(env_lengths[-1])
            logger.record_tabular("episode_total_return", episode_total_return)
            logger.record_tabular("episode_original_total_return", episode_original_total_return)
            logger.record_tabular("episode_length", episode_total_length)
            for metric_name in sorted(episode_custom_metrics.keys()):
                episode_metric_total = np.nan if len(episode_custom_metrics[metric_name]) == 0 else float(episode_custom_metrics[metric_name][-1])
                logger.record_tabular(metric_name, episode_metric_total)
            logger.dump_tabular()

            if wandb_run is not None:
                metrics = {
                    "steps": int(t),
                    "episodes": int(num_episodes),
                    "total_reward": float(reward_total),
                    "original_total_reward": float(original_reward_total),
                    "buffer_size": int(replay.size),
                    "epsilon": float(epsilon),
                    "loss": np.nan if last_loss is None else float(last_loss),
                    "episode_total_return": episode_total_return,
                    "episode_original_total_return": episode_original_total_return,
                    "episode_length": episode_total_length,
                }
                for metric_name in episode_custom_metrics.keys():
                    metrics[metric_name] = (
                        np.nan if len(episode_custom_metrics[metric_name]) == 0 else float(episode_custom_metrics[metric_name][-1])
                    )
                wandb_run.log(metrics, step=int(t))

            reward_total = 0.0
            original_reward_total = 0.0

    model = CleanDQNModel(q_net=q_net, params=params, hidden_dims=hidden_dims)

    if logger.get_dir() is not None:
        model_path = os.path.join(logger.get_dir(), "model.pkl")
        model.save(model_path)

    if wandb_run is not None:
        wandb_run.finish()

    return model
