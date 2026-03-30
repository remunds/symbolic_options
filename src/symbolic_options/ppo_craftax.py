import time
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax

import hydra
from omegaconf import OmegaConf
import wandb
from rtpt import RTPT

from symbolic_options.utils.video_recorder import video_callback

from craftax.craftax_env import make_craftax_env_from_name
from symbolic_options.purejaxql.craftax_wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    MultiRewardWrapper,
    MultiRewardLogWrapper,
    ExplorationMapWrapper
)
from symbolic_options.utils.video_recorder import CraftaxClassicRenderer
from symbolic_options.reward_functions.craftax_classic import survival_reward, combat_reward, resource_collection_reward, crafting_reward, explore, env_reward

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class CustomTrainState(TrainState):
    timesteps: int = 0
    n_updates: int = 0

rtpt = None
def rtpt_callback():
    global rtpt
    rtpt.step()

def make_train(config):
    global rtpt
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=f"{config['ALG_NAME']}_{config['ENV_NAME']}", max_iterations=config["NUM_UPDATES"])
    rtpt.start()
    renderer = CraftaxClassicRenderer()

    basic_env = make_craftax_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    ) 
    env_params = basic_env.default_params
    def create_env(env, train: bool = False, modif: bool = False):
        reward_funcs = [survival_reward, combat_reward, crafting_reward, resource_collection_reward, explore]
        env = ExplorationMapWrapper(env)
        env = MultiRewardWrapper(env, reward_funcs)
        env = MultiRewardLogWrapper(env)
        if train:
            if config["USE_OPTIMISTIC_RESETS"]:
                env = OptimisticResetVecEnvWrapper(
                    env,
                    num_envs=config["NUM_ENVS"],
                    reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]))
            else:
                env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])
        else:
            if config["USE_OPTIMISTIC_RESETS"]:
                env = OptimisticResetVecEnvWrapper(
                    env,
                    num_envs=config["TEST_NUM_ENVS"],
                    reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]))
            else:
                env = BatchEnvWrapper(env, num_envs=config["TEST_NUM_ENVS"])
        return env
    env = create_env(basic_env, train=True)
    test_env = create_env(basic_env, train=False, modif=False)
    test_env_modif = create_env(basic_env, train=False, modif=True)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        original_rng = rng[0]
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        # init_x = jnp.zeros(config["OBS_SHAPE"])
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        def get_test_metrics(train_state, modif, rng):
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = jax.lax.cond(
                modif,
                lambda _: test_env_modif.reset(_rng, env_params),
                lambda _: test_env.reset(_rng, env_params),
                operand=None
            )

            def _test_step(runner_state, unused):
                env_state, last_obs, rng = runner_state
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.mode()

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obs, env_state, reward, done, info = jax.lax.cond(
                    modif,
                    lambda _: test_env_modif.step(_rng, env_state, action, env_params),
                    lambda _: test_env.step(_rng, env_state, action, env_params),
                    operand=None
                )
                runner_state = (env_state, obs, rng)
                env_state_vid = jax.tree.map(lambda x: x[0], env_state)
                info.pop("all_rewards")  # Remove all_rewards from info
                return runner_state, (info, env_state_vid, done[0]) 

            rng, _rng = jax.random.split(rng)
            runner_state = (env_state, init_obs, _rng)
            runner_state, output = jax.lax.scan(
                _test_step, runner_state, None, config["TEST_NUM_STEPS"]
            )
            infos, states, dones = output

            # Aggregate metrics (if done)
            # test_metrics = {}
            # pre_str = "test_modif/" if modif else "test/"
            # for k, v in metrics.items():
            #     if isinstance(v, dict):
            #         for sub_k, sub_v in v.items():
            #             test_metrics[f"{pre_str}/{k}_{sub_k}"] = sub_v.mean()
            #     else:
            #         test_metrics[f"{pre_str}/{k}"] = v.mean()

            # Record video
            if config["RECORD_VIDEO"]:
                jax.lax.cond(
                    train_state.n_updates > 0,
                    lambda _: jax.debug.callback(video_callback, states, None, None, dones, train_state.n_updates, renderer, modif=modif),
                    lambda _: None,
                    operand=None,
                )

            # return mean of done infos
            done_infos = jax.tree.map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return done_infos

            # return test_metrics

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
                # info["all_rewards"] = jnp.squeeze(info["all_rewards"])
                info.pop("all_rewards")  # Remove all_rewards from info
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            train_state, env_state, last_obs, test_metrics, modif_metrics, rng = runner_state
            env_runner = (train_state, env_state, last_obs, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, env_runner, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng = runner_state

            # CALCULATE ADVANTAGE
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
            }

            infos = traj_batch.info
            metrics.update({k: v.mean() for k, v in infos.items()})

            rng = update_state[-1]
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"],
                n_updates=train_state.n_updates + 1,
            )

            # Get test metrics
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                train_state.timesteps % config["TEST_INTERVAL"] == 0,
                lambda ts, r: get_test_metrics(ts, False, r),
                # lambda ts, r: {k: jnp.array(jnp.nan) for k in [f"test/{mk}" for mk in infos.keys()]}, # Dummy zeros
                lambda ts, r: test_metrics,
                train_state,
                _rng
            )
            # metrics.update(test_metrics)
            metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            modif_metrics = jax.lax.cond(
                train_state.timesteps % config["TEST_INTERVAL"] == 0,
                lambda ts, r: get_test_metrics(ts, True, r),
                # lambda ts, r: {k: jnp.array(jnp.nan) for k in [f"modif/{mk}" for mk in infos.keys()]}, # Dummy zeros
                lambda ts, r: modif_metrics,
                train_state,
                _rng
            )

            # metrics.update(modif_metrics)
            metrics.update({f"test_modif/{k}": v for k, v in modif_metrics.items()})

            if config["WANDB_MODE"] != "disabled":
                def callback(metrics, original_rng):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_rng)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])
                jax.debug.callback(callback, metrics, original_rng)

            jax.debug.callback(rtpt_callback)
            
            runner_state = (train_state, env_state, last_obs, test_metrics, modif_metrics, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, False, _rng) 
        modif_metrics = get_test_metrics(train_state, True, _rng)
        runner_state = (train_state, env_state, obsv, test_metrics, modif_metrics, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))

    config = {**config, **config["alg"]}
    # config = {
    #     "SEED": 0,
    #     "NUM_SEEDS": 1,
    #     "LR": 2.5e-4,
    #     "NUM_ENVS": 128,
    #     "NUM_STEPS": 128,
    #     "TOTAL_TIMESTEPS": 5e7,
    #     "UPDATE_EPOCHS": 2,
    #     "NUM_MINIBATCHES": 4,
    #     "GAMMA": 0.99,
    #     "GAE_LAMBDA": 0.95,
    #     "CLIP_EPS": 0.2,
    #     "ENT_COEF": 0.01,
    #     "VF_COEF": 0.5,
    #     "MAX_GRAD_NORM": 0.5,
    #     "ACTIVATION": "relu",
    #     "ANNEAL_LR": True,
    #     "WANDB_MODE": "online",
    #     "ENTITY": "",
    #     "PROJECT": "",
    #     "ALG_NAME": "PPO",
    #     "ENV_NAME": "Kangaroo",
    #     "NAME_INITIALS": "RE",
    #     "TEST_INTERVAL": 10_000,
    #     "TEST_NUM_ENVS": 128,
    #     "TEST_NUM_STEPS": 10_000,
    #     "RECORD_VIDEO": True, 
    # }
    alg_name = config.get("ALG_NAME", "ppo")

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
    rngs = jax.random.split(rng, config["NUM_SEEDS"]) 
    print("Compiling...")
    start = time.time()
    train_jit = jax.jit(jax.vmap(make_train(config)))
    train_jit.lower(rngs).compile()
    print(f"Compilation took {time.time()-start} seconds.")
    outs = jax.block_until_ready(train_jit(rngs))
    print(f"Training took {time.time()-start} seconds.") 



if __name__ == "__main__":
    main(None)