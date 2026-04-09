import copy
import os
import time
from typing import Any, NamedTuple, Sequence

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from purejaxql.brax_wrappers import (
    ClipAction,
    LogVecWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    PlaygroundVecGymnaxWrapper,
    get_original_state,
)
from purejaxql.save_load import save


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray
    hidden_sizes: Sequence[int]
    activation: str = "relu"
    norm_type: str = "layer_norm"
    norm_input: bool = False
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=False):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        else:
            # Keep batch norm state shape stable when input normalization is disabled.
            _ = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda y: nn.LayerNorm(epsilon=1e-6)(y)
        elif self.norm_type == "batch_norm":
            normalize = lambda y: nn.BatchNorm(
                use_running_average=not train, epsilon=1e-5
            )(y)
        else:
            normalize = lambda y: y

        for hs in self.hidden_sizes:
            x = nn.Dense(hs, kernel_init=nn.initializers.orthogonal(self.init_scale))(x)
            x = normalize(x)
            x = activation(x)

        x = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale))(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class Critic(nn.Module):
    hidden_sizes: Sequence[int]
    norm_type: str = "layer_norm"
    norm_input: bool = False
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x, action, train=False):
        x = jnp.concatenate([x, action], axis=-1)

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        else:
            # Keep batch norm state shape stable when input normalization is disabled.
            _ = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda y: nn.LayerNorm(epsilon=1e-6)(y)
        elif self.norm_type == "batch_norm":
            normalize = lambda y: nn.BatchNorm(
                use_running_average=not train, epsilon=1e-5
            )(y)
        else:
            normalize = lambda y: y

        for hs in self.hidden_sizes:
            x = nn.Dense(hs, kernel_init=nn.initializers.orthogonal(self.init_scale))(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(1, kernel_init=nn.initializers.orthogonal(self.init_scale))(x)
        return jnp.squeeze(x, axis=-1)


class MetaQNetwork(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int]
    activation: str = "relu"
    norm_type: str = "layer_norm"
    norm_input: bool = False
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x, train=False):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)
        else:
            # Keep batch norm state shape stable when input normalization is disabled.
            _ = nn.BatchNorm(use_running_average=not train, epsilon=1e-5)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda y: nn.LayerNorm(epsilon=1e-6)(y)
        elif self.norm_type == "batch_norm":
            normalize = lambda y: nn.BatchNorm(
                use_running_average=not train, epsilon=1e-5
            )(y)
        else:
            normalize = lambda y: y

        for hs in self.hidden_sizes:
            x = nn.Dense(hs, kernel_init=nn.initializers.orthogonal(self.init_scale))(x)
            x = normalize(x)
            x = activation(x)

        x = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale))(x)
        return x


class OptionTransition(NamedTuple):
    done: jnp.ndarray
    active_agent: jnp.ndarray
    original_action: jnp.ndarray
    action: jnp.ndarray
    option_values: jnp.ndarray
    option_rewards: jnp.ndarray
    meta_q_values: jnp.ndarray
    env_reward: jnp.ndarray
    obs: Any
    next_obs: Any
    info: Any


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def smooth_l1_loss(pred, target, beta=1.0):
    diff = pred - target
    abs_diff = jnp.abs(diff)
    loss = jnp.where(
        abs_diff < beta,
        0.5 * (diff**2) / beta,
        abs_diff - 0.5 * beta,
    )
    return loss


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    num_agents = config.get("NUM_AGENTS", 1)
    assert num_agents > 0, "NUM_AGENTS must be > 0"

    env, env_params = PlaygroundVecGymnaxWrapper(config["ENV_NAME"]), None
    print("episode_length:", env.episode_length)

    action_space = env.action_space(None)
    env = LogVecWrapper(env)
    env = ClipAction(env, low=action_space.low, high=action_space.high)

    if config["NORMALIZE_REWARD"]:
        env = NormalizeVecReward(env, config["GAMMA"])
    if config["NORMALIZE_OBS"]:
        env = NormalizeVecObservation(env)

    if config.get("TEST_DURING_TRAINING", True):
        config["TEST_NUM_STEPS"] = env.episode_length
        config["TEST_NUM_ENVS"] = (
            config["TEST_NUM_ENVS"]
            if config["TEST_NUM_ENVS"] is not None
            else config["NUM_ENVS"]
        )
        print("Test num steps:", config["TEST_NUM_STEPS"])
        print("Test num envs:", config["TEST_NUM_ENVS"])

    lr_scheduler = optax.linear_schedule(
        init_value=config["LR_START"],
        end_value=config["LR_END"],
        transition_steps=(config["NUM_UPDATES_DECAY"] * config["LR_DECAY"])
        * config["NUM_MINIBATCHES"]
        * config["NUM_EPOCHS"],
    )
    lr = lr_scheduler if config.get("ANNEAL_LR", False) else config["LR_START"]

    noise_scheduler = optax.linear_schedule(
        init_value=config["NOISE_START"],
        end_value=config["NOISE_FINISH"],
        transition_steps=(config["NOISE_DECAY"] * config["NUM_UPDATES_DECAY"]),
    )

    eps_meta_scheduler = optax.linear_schedule(
        init_value=config["META_EPS_START"],
        end_value=config["META_EPS_FINISH"],
        transition_steps=(config["META_EPS_DECAY"] * config["NUM_UPDATES_DECAY"]),
    )

    log_times = []

    def log_eval_video(env_state_traj, step):
        if config["WANDB_MODE"] == "disabled" or not config.get("RECORD_VIDEO", False):
            return

        os.environ.setdefault("MUJOCO_GL", config.get("MUJOCO_GL", "egl"))
        num_frames = min(config.get("VIDEO_NUM_STEPS", 256), config["TEST_NUM_STEPS"])
        if num_frames <= 0:
            return

        raw_state_traj = get_original_state(env_state_traj)
        trajectory = [
            jax.tree_util.tree_map(
                lambda x, t=t: np.asarray(x[t, 0]),
                raw_state_traj,
            )
            for t in range(num_frames)
        ]

        frames = env._env.render(
            trajectory,
            height=config.get("VIDEO_HEIGHT", 256),
            width=config.get("VIDEO_WIDTH", 256),
            camera=config.get("VIDEO_CAMERA", None),
        )
        frames = np.asarray(frames, dtype=np.uint8)
        if frames.size == 0:
            return

        wandb.log(
            {
                "test/video": wandb.Video(
                    frames,
                    fps=config.get("VIDEO_FPS", 20),
                    format="mp4",
                ),
                "test/video_update": int(step),
            }
        )

    def train(rng):
        original_rng = rng[0]

        actor = Actor(
            env.action_space(env_params).shape[0],
            action_scale=jnp.array((env.high - env.low) / 2.0),
            action_bias=jnp.array((env.high + env.low) / 2.0),
            hidden_sizes=config["ACTOR_HIDDEN_SIZES"],
            activation=config.get("ACTIVATION", "relu"),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            init_scale=config.get("ACTOR_INIT_SCALE", 1.0),
        )

        critic = Critic(
            hidden_sizes=config["CRITIC_HIDDEN_SIZES"],
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            init_scale=config.get("CRITIC_INIT_SCALE", 1.0),
        )

        meta_network = MetaQNetwork(
            action_dim=num_agents,
            hidden_sizes=config.get("META_HIDDEN_SIZES", config["CRITIC_HIDDEN_SIZES"]),
            activation=config.get("ACTIVATION", "relu"),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            init_scale=config.get("META_INIT_SCALE", 1.0),
        )

        init_actor_x = jnp.zeros(env.observation_space(env_params)["actor"].shape)
        init_critic_x = jnp.zeros(env.observation_space(env_params)["critic"].shape)
        dummy_action = jnp.zeros(env.action_size)

        def init_option_states(rng_key):
            rng_key, actor_key, critic_key = jax.random.split(rng_key, 3)
            actor_variables = actor.init(actor_key, init_actor_x)
            critic_keys = jax.random.split(critic_key, config["NUM_CRITICS"])
            critic_variables = jax.vmap(critic.init, in_axes=(0, None, None))(
                critic_keys, init_critic_x, dummy_action
            )
            return actor_variables, critic_variables

        rng, init_key = jax.random.split(rng)
        option_keys = jax.random.split(init_key, num_agents)
        actor_variables, critic_variables = jax.vmap(init_option_states)(option_keys)

        rng, meta_key = jax.random.split(rng)
        meta_variables = meta_network.init(meta_key, init_actor_x)

        tx_actor = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )
        tx_critic = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )
        tx_meta = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.radam(learning_rate=lr),
        )

        def create_option_train_states(actor_vars, critic_vars):
            train_state_actor = CustomTrainState.create(
                apply_fn=actor.apply,
                params=actor_vars["params"],
                batch_stats=actor_vars["batch_stats"],
                tx=tx_actor,
            )
            train_state_critic = CustomTrainState.create(
                apply_fn=critic.apply,
                params=critic_vars["params"],
                batch_stats=critic_vars["batch_stats"],
                tx=tx_critic,
            )
            return train_state_actor, train_state_critic

        train_state_actor, train_state_critic = jax.vmap(create_option_train_states)(
            actor_variables, critic_variables
        )

        train_state_meta = CustomTrainState.create(
            apply_fn=meta_network.apply,
            params=meta_variables["params"],
            batch_stats=meta_variables["batch_stats"],
            tx=tx_meta,
        )

        train_state = {
            "actor": train_state_actor,
            "critic": train_state_critic,
            "meta": train_state_meta,
        }

        def actor_critic_step(train_state, obs, rng_key, noise_std=0.0, num_envs=None):
            if num_envs is None:
                num_envs = config["NUM_ENVS"]

            actor_state = train_state["actor"]
            critic_state = train_state["critic"]

            def single_actor_step(params, batch_stats):
                action = actor.apply(
                    {
                        "params": params,
                        "batch_stats": batch_stats,
                    },
                    obs["actor"],
                    train=False,
                )
                return action

            actions = jax.vmap(single_actor_step)(
                actor_state.params,
                actor_state.batch_stats,
            )
            original_actions = actions.copy()

            rng_key, noise_key = jax.random.split(rng_key)

            if config.get("LINSPACE_NOISE", False):
                noise_stds = jnp.linspace(0, noise_std, num_envs)
            else:
                noise_stds = jnp.full((num_envs,), noise_std)

            action_scale = jnp.broadcast_to(
                jnp.asarray(actor.action_scale),
                (actions.shape[-1],),
            )

            noise = (
                jax.random.normal(noise_key, actions.shape)
                * noise_stds[None, :, None]
                * action_scale[None, None, :]
            )
            actions = actions + noise
            actions = env.clip_action(actions)

            def single_option_value(critic_params, critic_batch_stats, option_action):
                def single_critic_value(params, bstats):
                    return critic.apply(
                        {
                            "params": params,
                            "batch_stats": bstats,
                        },
                        obs["critic"],
                        option_action,
                        train=False,
                    )

                values = jax.vmap(single_critic_value)(critic_params, critic_batch_stats)
                return jnp.mean(values, axis=0)

            option_values = jax.vmap(single_option_value)(
                critic_state.params,
                critic_state.batch_stats,
                actions,
            )

            return original_actions, actions, option_values, noise

        def meta_step(train_state_meta, obs):
            return meta_network.apply(
                {
                    "params": train_state_meta.params,
                    "batch_stats": train_state_meta.batch_stats,
                },
                obs["actor"],
                train=False,
            )

        def eps_greedy_meta(rng_key, q_vals, eps):
            rng_a, rng_e = jax.random.split(rng_key)
            greedy = jnp.argmax(q_vals, axis=-1)
            random_actions = jax.random.randint(
                rng_a,
                shape=greedy.shape,
                minval=0,
                maxval=q_vals.shape[-1],
            )
            pick_random = jax.random.uniform(rng_e, greedy.shape) < eps
            return jnp.where(pick_random, random_actions, greedy)

        def get_option_rewards(info, reward):
            if config.get("USE_OPTION_REWARDS", False):
                return info[config.get("OPTION_REWARD_KEY", "all_rewards")][:, :num_agents]
            return jnp.repeat(reward[:, None], num_agents, axis=1)

        def update_single_option(
            option_idx,
            actor_state,
            critic_state,
            transitions,
            lambda_targets,
            rng_key,
        ):
            timesteps = actor_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]

            def preprocess_transition(x, prep_rng):
                x = x.reshape(-1, *x.shape[2:])
                x = jax.random.permutation(prep_rng, x)
                x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                return x

            def _learn_phase(carry, minibatch_and_target):
                actor_state_i, critic_state_i, phase_rng = carry
                minibatch, target = minibatch_and_target

                def _critic_loss_fn(critic_params):
                    def single_critic_pass(params, batch_stats):
                        values, updates = critic.apply(
                            {
                                "params": params,
                                "batch_stats": batch_stats,
                            },
                            minibatch.obs["critic"],
                            minibatch.action,
                            train=True,
                            mutable=["batch_stats"],
                        )
                        return values, updates

                    values, updates = jax.vmap(single_critic_pass)(
                        critic_params, critic_state_i.batch_stats
                    )

                    value_losses = jax.vmap(smooth_l1_loss, in_axes=(0, None))(
                        values, target
                    )
                    losses = jax.vmap(lambda z: jnp.mean(z))(value_losses)
                    loss = jnp.sum(losses)

                    value_diff = jnp.abs(values - minibatch.option_values[:, option_idx])
                    loss_infos = {
                        "critic_value_loss": value_losses.mean(),
                        "critic_value_diff": value_diff.mean(),
                    }
                    return loss, (updates, loss_infos)

                def _actor_loss_fn(actor_params):
                    action, updates = actor.apply(
                        {
                            "params": actor_params,
                            "batch_stats": actor_state_i.batch_stats,
                        },
                        minibatch.obs["actor"],
                        train=True,
                        mutable=["batch_stats"],
                    )

                    def single_critic_value(params, batch_stats):
                        return critic.apply(
                            {
                                "params": params,
                                "batch_stats": batch_stats,
                            },
                            minibatch.obs["critic"],
                            action,
                            train=False,
                        )

                    values = jax.vmap(single_critic_value)(
                        critic_state_i.params,
                        critic_state_i.batch_stats,
                    )
                    rl_loss = jnp.mean(values, axis=0)

                    action_diff = action - minibatch.original_action
                    action_diff = (action_diff - actor.action_bias) / actor.action_scale
                    action_diff = jnp.abs(action_diff).mean(axis=-1)
                    pen_loss = smooth_l1_loss(action, minibatch.original_action).mean(axis=-1)

                    penalty = jnp.where(
                        action_diff < config["THRESHOLD"],
                        0.0,
                        config["PENALTY_COEFF"] * pen_loss,
                    )
                    actor_loss = jnp.mean(-rl_loss + penalty)

                    loss_infos = {
                        "policy_loss": rl_loss.mean(),
                        "actor_penalty_loss": pen_loss.mean(),
                        "actor_penalty_ratio": (
                            action_diff > config["PENALTY_COEFF"]
                        ).mean(),
                        "actor_penalty_diff": action_diff.mean(),
                    }
                    return actor_loss, (updates, loss_infos)

                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                (
                    critic_loss,
                    (batch_stats_update_critic, critic_loss_infos),
                ), critic_grads = critic_grad_fn(critic_state_i.params)
                critic_state_i = critic_state_i.apply_gradients(grads=critic_grads)
                critic_state_i = critic_state_i.replace(
                    grad_steps=critic_state_i.grad_steps + 1,
                    batch_stats=batch_stats_update_critic["batch_stats"],
                )

                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                (
                    actor_loss,
                    (batch_stats_update_actor, actor_loss_infos),
                ), actor_grads = actor_grad_fn(actor_state_i.params)
                actor_state_i = actor_state_i.apply_gradients(grads=actor_grads)
                actor_state_i = actor_state_i.replace(
                    grad_steps=actor_state_i.grad_steps + 1,
                    batch_stats=batch_stats_update_actor["batch_stats"],
                )

                total_loss = actor_loss + critic_loss
                opt_infos = {
                    "critic_norm": optax.global_norm(critic_grads),
                    "actor_norm": optax.global_norm(actor_grads),
                }
                opt_infos.update(critic_loss_infos)
                opt_infos.update(actor_loss_infos)
                return (actor_state_i, critic_state_i, phase_rng), (
                    total_loss,
                    critic_loss,
                    actor_loss,
                    opt_infos,
                )

            def _learn_epoch(carry, _):
                actor_state_i, critic_state_i, epoch_rng = carry
                epoch_rng, prep_rng = jax.random.split(epoch_rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, prep_rng), transitions
                )
                targets = preprocess_transition(lambda_targets, prep_rng)

                (actor_state_i, critic_state_i, epoch_rng), loss_infos = jax.lax.scan(
                    _learn_phase, (actor_state_i, critic_state_i, epoch_rng), (minibatches, targets)
                )
                return (actor_state_i, critic_state_i, epoch_rng), loss_infos

            rng_key, learn_rng = jax.random.split(rng_key)
            (
                actor_state,
                critic_state,
                _,
            ), (
                total_loss,
                critic_loss,
                actor_loss,
                opt_infos,
            ) = jax.lax.scan(
                _learn_epoch,
                (actor_state, critic_state, learn_rng),
                None,
                config["NUM_EPOCHS"],
            )

            actor_state = actor_state.replace(
                timesteps=timesteps,
                n_updates=actor_state.n_updates + 1,
            )
            critic_state = critic_state.replace(
                timesteps=timesteps,
                n_updates=critic_state.n_updates + 1,
            )

            metrics = {
                "loss": total_loss.mean(),
                "value_loss": critic_loss.mean(),
                "loss_actor": actor_loss.mean(),
                "grad_steps_actor": actor_state.grad_steps,
                "grad_steps_critic": critic_state.grad_steps,
                "noise": noise_scheduler(actor_state.n_updates),
                "lr": lr_scheduler(actor_state.n_updates),
                "critic_norm": opt_infos["critic_norm"].mean(),
                "actor_norm": opt_infos["actor_norm"].mean(),
                "policy_loss": opt_infos["policy_loss"].mean(),
                "actor_penalty_loss": opt_infos["actor_penalty_loss"].mean(),
                "actor_penalty_ratio": opt_infos["actor_penalty_ratio"].mean(),
                "actor_penalty_diff": opt_infos["actor_penalty_diff"].mean(),
                "critic_value_loss": opt_infos["critic_value_loss"].mean(),
                "critic_value_diff": opt_infos["critic_value_diff"].mean(),
            }
            return actor_state, critic_state, metrics

        def update_meta_network(meta_state, transitions, last_meta_q, rng_key):
            timesteps = meta_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                    transition.env_reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                    target_bootstrap + config["GAMMA"] * config["META_LAMBDA"] * delta
                )
                lambda_returns = (
                    (1 - transition.done) * lambda_returns
                    + transition.done * transition.env_reward
                )
                next_q = jnp.max(transition.meta_q_values, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            bootstrap_next_q = jnp.max(last_meta_q, axis=-1) * (1 - transitions.done[-1])
            lambda_returns = transitions.env_reward[-1] + config["GAMMA"] * bootstrap_next_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, bootstrap_next_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

            def preprocess_transition(x, prep_rng):
                x = x.reshape(-1, *x.shape[2:])
                x = jax.random.permutation(prep_rng, x)
                x = x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                return x

            def _learn_phase(carry, minibatch_and_target):
                meta_state_i, phase_rng = carry
                minibatch, target = minibatch_and_target

                def _loss_fn(meta_params):
                    q_vals, updates = meta_network.apply(
                        {
                            "params": meta_params,
                            "batch_stats": meta_state_i.batch_stats,
                        },
                        minibatch.obs["actor"],
                        train=True,
                        mutable=["batch_stats"],
                    )
                    chosen_q = jnp.take_along_axis(
                        q_vals,
                        minibatch.active_agent[:, None],
                        axis=-1,
                    ).squeeze(axis=-1)
                    loss = 0.5 * jnp.square(chosen_q - target).mean()
                    return loss, (updates, chosen_q)

                (loss, (updates, chosen_q)), grads = jax.value_and_grad(
                    _loss_fn, has_aux=True
                )(meta_state_i.params)
                meta_state_i = meta_state_i.apply_gradients(grads=grads)
                meta_state_i = meta_state_i.replace(
                    grad_steps=meta_state_i.grad_steps + 1,
                    batch_stats=updates["batch_stats"],
                )
                return (meta_state_i, phase_rng), (loss, chosen_q)

            def _learn_epoch(carry, _):
                meta_state_i, epoch_rng = carry
                epoch_rng, prep_rng = jax.random.split(epoch_rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, prep_rng), transitions
                )
                targets = preprocess_transition(lambda_targets, prep_rng)

                (meta_state_i, epoch_rng), (loss, chosen_q) = jax.lax.scan(
                    _learn_phase, (meta_state_i, epoch_rng), (minibatches, targets)
                )
                return (meta_state_i, epoch_rng), (loss, chosen_q)

            rng_key, learn_rng = jax.random.split(rng_key)
            (meta_state, _), (loss, chosen_q) = jax.lax.scan(
                _learn_epoch,
                (meta_state, learn_rng),
                None,
                config["NUM_EPOCHS"],
            )

            meta_state = meta_state.replace(
                timesteps=timesteps,
                n_updates=meta_state.n_updates + 1,
            )

            metrics = {
                "meta_loss": loss.mean(),
                "meta_qvals": chosen_q.mean(),
                "meta_eps": eps_meta_scheduler(meta_state.n_updates),
                "meta_grad_steps": meta_state.grad_steps,
            }
            return meta_state, metrics

        def get_test_metrics(train_state, training_env_state, rng_key):
            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, inner_rng, returns = carry
                inner_rng, ac_rng, meta_rng, step_rng = jax.random.split(inner_rng, 4)

                _, option_actions, _, _ = actor_critic_step(
                    train_state,
                    last_obs,
                    ac_rng,
                    noise_std=config.get("TEST_NOISE", 0.0),
                    num_envs=config["TEST_NUM_ENVS"],
                )
                meta_q = meta_step(train_state["meta"], last_obs)

                eps = jnp.full((config["TEST_NUM_ENVS"],), config.get("EPS_TEST", 0.0))
                active_agent = eps_greedy_meta(meta_rng, meta_q, eps)
                env_indices = jnp.arange(config["TEST_NUM_ENVS"])
                action = option_actions[active_agent, env_indices]

                step_rngs = jax.random.split(step_rng, config["TEST_NUM_ENVS"])
                new_obs, new_env_state, reward, done, info = env.step(
                    step_rngs, env_state, action, env_params
                )

                reward_key = config.get("TEST_RETURN_KEY", "original_reward")
                reward_for_return = info[reward_key] if reward_key in info else reward

                returns["running_returns"] = jnp.where(
                    ~returns["running_done"],
                    returns["running_returns"] + reward_for_return,
                    returns["running_returns"],
                )
                returns["running_len"] = jnp.where(
                    ~returns["running_done"],
                    returns["running_len"] + 1,
                    returns["running_len"],
                )
                returns["running_done"] = (returns["running_done"] + done).astype(bool)

                return (new_env_state, new_obs, inner_rng, returns), new_env_state

            rng_key, reset_rng = jax.random.split(rng_key)
            reset_rngs = jax.random.split(reset_rng, config["TEST_NUM_ENVS"])
            init_obs, reset_env_state = env.reset(reset_rngs, env_params)

            if config["NORMALIZE_OBS"]:
                env_state = training_env_state.replace(env_state=reset_env_state.env_state)
                init_obs["actor"] = (
                    init_obs["actor"] - env_state.actor_mean
                ) / jnp.sqrt(env_state.actor_var + 1e-8)
            else:
                env_state = reset_env_state

            returns = {
                "running_returns": jnp.zeros((config["TEST_NUM_ENVS"],)),
                "running_len": jnp.zeros((config["TEST_NUM_ENVS"],)),
                "running_done": jnp.zeros((config["TEST_NUM_ENVS"],), dtype=bool),
            }

            (_, _, _, returns), env_state_traj = jax.lax.scan(
                _env_step,
                (env_state, init_obs, rng_key, returns),
                None,
                config["TEST_NUM_STEPS"],
            )

            if config.get("RECORD_VIDEO", False):
                jax.debug.callback(
                    log_eval_video,
                    env_state_traj,
                    train_state["meta"].n_updates,
                )

            done_infos = {
                "returned_episode_returns": returns["running_returns"].sum()
                / jnp.maximum(returns["running_done"].sum(), 1),
                "returned_episode_lengths": returns["running_len"].sum()
                / jnp.maximum(returns["running_done"].sum(), 1),
                "done_episodes": returns["running_done"].sum()
                / config["TEST_NUM_ENVS"],
            }
            return done_infos

        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng_key, test_metrics = runner_state

            def _env_step(carry, _):
                train_state_i, env_state_i, obs_i, inner_rng, carry_aux = carry
                inner_rng, ac_rng, meta_rng, step_rng = jax.random.split(inner_rng, 4)

                noise_std = noise_scheduler(train_state_i["meta"].n_updates)
                original_actions, option_actions, option_values, _ = actor_critic_step(
                    train_state_i,
                    obs_i,
                    ac_rng,
                    noise_std,
                    num_envs=config["NUM_ENVS"],
                )

                meta_q = meta_step(train_state_i["meta"], obs_i)
                eps_meta = jnp.full(
                    (config["NUM_ENVS"],),
                    eps_meta_scheduler(train_state_i["meta"].n_updates),
                )
                active_agent = eps_greedy_meta(meta_rng, meta_q, eps_meta)

                env_indices = jnp.arange(config["NUM_ENVS"])
                action = option_actions[active_agent, env_indices]
                original_action = original_actions[active_agent, env_indices]

                step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
                new_obs, new_env_state, reward, done, info = env.step(
                    step_rngs, env_state_i, action, env_params
                )

                option_rewards = get_option_rewards(info, reward)
                transition = OptionTransition(
                    done=done,
                    active_agent=active_agent,
                    original_action=original_action,
                    action=action,
                    option_values=jnp.swapaxes(option_values, 0, 1),
                    option_rewards=option_rewards,
                    meta_q_values=meta_q,
                    env_reward=reward,
                    obs=obs_i,
                    next_obs=new_obs,
                    info=info,
                )

                return (train_state_i, new_env_state, new_obs, inner_rng, carry_aux), transition

            (train_state, env_state, last_obs, rng_key, _), traj_batch = jax.lax.scan(
                _env_step,
                (train_state, env_state, last_obs, rng_key, test_metrics),
                None,
                config["NUM_STEPS"],
            )

            rng_key, bootstrap_rng = jax.random.split(rng_key)
            _, _, last_option_values, _ = actor_critic_step(
                train_state,
                last_obs,
                bootstrap_rng,
                noise_std=0.0,
                num_envs=config["NUM_ENVS"],
            )
            last_option_values = jnp.swapaxes(last_option_values, 0, 1)

            rng_key, _ = jax.random.split(rng_key)
            last_meta_q = meta_step(train_state["meta"], last_obs)

            def build_lambda_targets(option_idx):
                option_rewards = traj_batch.option_rewards[..., option_idx]
                option_values = traj_batch.option_values[..., option_idx]
                last_val = last_option_values[:, option_idx] * (1 - traj_batch.done[-1])
                lambda_returns = option_rewards[-1] + config["GAMMA"] * last_val

                def _get_target(carry, transition):
                    lambda_ret, next_q = carry
                    target_bootstrap = (
                        transition.option_rewards[:, option_idx]
                        + config["GAMMA"] * (1 - transition.done) * next_q
                    )
                    delta = lambda_ret - next_q
                    lambda_ret = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_ret = (
                        (1 - transition.done) * lambda_ret
                        + transition.done * transition.option_rewards[:, option_idx]
                    )
                    next_q = transition.option_values[:, option_idx]
                    return (lambda_ret, next_q), lambda_ret

                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_val),
                    jax.tree_util.tree_map(lambda x: x[:-1], traj_batch),
                    reverse=True,
                )
                return jnp.concatenate((targets, lambda_returns[jnp.newaxis]))

            option_indices = jnp.arange(num_agents)
            lambda_targets = jax.vmap(build_lambda_targets)(option_indices)

            rng_key, option_update_rng = jax.random.split(rng_key)
            option_rngs = jax.random.split(option_update_rng, num_agents)

            actor_states, critic_states, option_metrics = jax.vmap(
                update_single_option,
                in_axes=(
                    0,
                    0,
                    0,
                    None,
                    0,
                    0,
                ),
            )(
                option_indices,
                train_state["actor"],
                train_state["critic"],
                traj_batch,
                lambda_targets,
                option_rngs,
            )

            train_state = {
                "actor": actor_states,
                "critic": critic_states,
                "meta": train_state["meta"],
            }

            rng_key, meta_update_rng = jax.random.split(rng_key)
            updated_meta_state, meta_metrics = update_meta_network(
                train_state["meta"],
                traj_batch,
                last_meta_q,
                meta_update_rng,
            )
            train_state = {
                "actor": train_state["actor"],
                "critic": train_state["critic"],
                "meta": updated_meta_state,
            }

            env_step = train_state["meta"].timesteps
            update_steps = train_state["meta"].n_updates

            metrics = {
                "env_step": env_step,
                "update_steps": update_steps,
                "env_frame": env_step,
                "noise": noise_scheduler(update_steps),
                "lr": lr_scheduler(update_steps),
            }
            for k, v in meta_metrics.items():
                metrics[k] = v

            for k, v in option_metrics.items():
                for i in range(v.shape[0]):
                    metrics[f"option_{i}/{k}"] = v[i]

            info_metrics = {}
            for k, v in traj_batch.info.items():
                if isinstance(v, jax.Array):
                    info_metrics[k] = v.mean()
            metrics.update(info_metrics)

            if config.get("TEST_DURING_TRAINING", False):
                rng_key, test_rng = jax.random.split(rng_key)
                test_metrics = jax.lax.cond(
                    train_state["meta"].n_updates
                    % jnp.maximum(
                        int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]),
                        1,
                    )
                    == 0,
                    lambda _: get_test_metrics(train_state, env_state, test_rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            if config["WANDB_MODE"] != "disabled":

                def callback(log_metrics, rng_value):
                    log_times.append(time.time())
                    if len(log_times) > 1:
                        dt = log_times[-1] - log_times[-2]
                        steps_per_update = (
                            config["NUM_ENVS"]
                            * config["NUM_STEPS"]
                            * config["NUM_SEEDS"]
                        )
                        log_metrics["sps"] = steps_per_update / dt
                        log_metrics["walltime"] = log_times[-1] - log_times[0]
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        log_metrics.update(
                            {
                                f"rng{int(rng_value)}/{k}": v
                                for k, v in log_metrics.items()
                            }
                        )
                    for k, v in log_metrics.items():
                        if np.isnan(v):
                            print(f"Warning: {k} contains NaN values")
                        if np.isinf(v):
                            print(f"Warning: {k} contains Inf values")
                    wandb.log(log_metrics)

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (train_state, env_state, last_obs, rng_key, test_metrics)
            return_metrics = metrics if config.get("RETURN_METRICS", False) else None
            return runner_state, return_metrics

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rngs, env_params)

        rng, test_rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, env_state, test_rng)

        rng, loop_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, loop_rng, test_metrics)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):
    config = {**config, **config["alg"]}

    alg_name = f'{config.get("ALG_NAME", "hierarchical_pqn")}_seed{config["SEED"]}'
    env_name = config["ENV_NAME"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["ALG_NAME"]}_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    t0 = time.time()
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    outs = jax.block_until_ready(train_vjit(rngs))
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        train_state = outs["runner_state"][0]
        env_state = outs["runner_state"][1]
        to_save = {
            "actor": {
                "params": train_state["actor"].params,
                "batch_stats": train_state["actor"].batch_stats,
            },
            "critic": {
                "params": train_state["critic"].params,
                "batch_stats": train_state["critic"].batch_stats,
            },
            "meta": {
                "params": train_state["meta"].params,
                "batch_stats": train_state["meta"].batch_stats,
            },
        }

        if config["NORMALIZE_OBS"] and hasattr(env_state, "actor_mean"):
            normalization_stats = {
                "actor_mean": env_state.actor_mean,
                "actor_var": env_state.actor_var,
                "critic_mean": env_state.critic_mean,
                "critic_var": env_state.critic_var,
            }
            to_save["normalization_stats"] = normalization_stats

        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        save_name = f'{config["ALG_NAME"]}_{config["ENV_NAME"]}_seed{config["SEED"]}'
        save(to_save, config, save_dir, save_name, vmaps=config["NUM_SEEDS"])


def tune(default_config):
    default_config = {**default_config, **default_config["alg"]}
    alg_name = default_config.get("ALG_NAME", "hierarchical_pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

            print("running experiment with params:", config)

            rng = jax.random.PRNGKey(config["SEED"])
            rngs = jax.random.split(rng, config["NUM_SEEDS"])
            train_vjit = jax.jit(jax.vmap(make_train(config)))
            _ = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "grid",
        "metric": {
            "name": "test/returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR_START": {
                "min": 1e-5,
                "max": 1e-3,
            },
            "META_EPS_FINISH": {
                "values": [0.01, 0.05, 0.1],
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config,
        entity=default_config["ENTITY"],
        project=default_config["PROJECT"],
    )
    wandb.agent(
        sweep_id,
        function=wrapped_make_train,
        entity=default_config["ENTITY"],
        project=default_config["PROJECT"],
        count=1000,
    )


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
