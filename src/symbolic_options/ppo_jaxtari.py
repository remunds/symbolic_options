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
from jaxatari.wrappers import MultiRewardLogWrapper
from jaxatari.wrappers import ObjectCentricWrapper, FlattenObservationWrapper, AtariWrapper

from symbolic_options.utils.video_recorder import video_callback

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
    
    if config.get("ENV_NAME", None) == "Pong":
        from jaxatari.games.jax_pong import JaxPong, PongRenderer
        from jaxatari.wrappers import MultiRewardLogWrapper
        from symbolic_options.reward_functions.pong import track_and_align, return_shot, defensive_positioning, llm_meta_policy, combined_meta_policy, learned_meta_policy, conditional_meta_policy, env_reward 
        from jaxatari.games.mods.pong_mods import LazyEnemyWrapper, RandomizedEnemyWrapper

        reward_funcs = [track_and_align, return_shot, defensive_positioning]
        if config.get("NO_REWARDS", False):
            reward_funcs = [env_reward, env_reward, env_reward] #use env_reward for all options

        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        noisy_training = config.get("NOISY_TRAINING", False)
        noisy_evaluation = config.get("NOISY_EVAL", False)
        detection_probability = config.get("DETECT_PROB", 1.0)
        noise_std_dev = config.get("NOISE_STD_DEV", 0.0)

        def create_env(train: bool = False, randomized_enemy: bool = False, lazy_enemy: bool = False):
            env = JaxPong(reward_funcs=reward_funcs)
            if randomized_enemy:
                env = RandomizedEnemyWrapper(env)
            if lazy_enemy:
                env = LazyEnemyWrapper(env)
            if train:
                if noisy_training:
                    env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life, detect_prob=detection_probability, std_dev=noise_std_dev)
                else:
                    env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
            else:
                if noisy_evaluation:
                    env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=detection_probability, std_dev=noise_std_dev)
                else:
                    env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False, False) 
        test_env = create_env(False, False, False) 
        test_env_modif = create_env(False, False, True) # evaluate on lazy enemy 
        renderer = PongRenderer()

    elif config.get("ENV_NAME", None) == "Breakout":
        from jaxatari.games.jax_breakout import JaxBreakout, BreakoutRenderer
        from jaxatari.wrappers import MultiRewardLogWrapper
        from symbolic_options.reward_functions.breakout import track_and_align, return_shot, defensive_positioning, llm_meta_policy, combined_meta_policy, learned_meta_policy, conditional_meta_policy, env_reward 
        from jaxatari.games.mods.breakout_mods import SpeedMode, SmallPaddle, BigPaddle

        reward_funcs = [track_and_align, return_shot, defensive_positioning]
        if config.get("NO_REWARDS", False):
            reward_funcs = [env_reward, env_reward, env_reward] #use env_reward for all options

        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        def create_env(train: bool = False, left_drift: bool = False, small_paddle: bool = False):
            env = JaxBreakout(reward_funcs=reward_funcs)
            if left_drift:
                env = SpeedMode(env)
            if small_paddle:
                env = BigPaddle(env)
            if train:
                env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False, False)
        test_env = create_env(False, False, False)
        test_env_modif = create_env(False, False, False) # evaluate on mod
        renderer = BreakoutRenderer()

    elif config.get("ENV_NAME", None) == "Freeway":
        from jaxatari.games.jax_freeway import JaxFreeway, FreewayRenderer
        from jaxatari.wrappers import MultiRewardLogWrapper
        from symbolic_options.reward_functions.freeway import avoid_crash, go_forward, llm_meta_policy, combined_meta_policy, learned_meta_policy, conditional_meta_policy, env_reward 
        from jaxatari.games.mods.freeway_mods import SpeedMode, StopAllCars, AlwaysStopAllCars

        reward_funcs = [avoid_crash, go_forward]
        if config.get("NO_REWARDS", False):
            reward_funcs = [env_reward, env_reward] #use env_reward for all options

        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        def create_env(train: bool = False, stop_all_cars: bool = False, stop_random_cars: bool = False, speed_mode: bool = False): 
            env = JaxFreeway(reward_funcs=reward_funcs)
            if stop_all_cars:
                env = AlwaysStopAllCars(env)
            if stop_random_cars:
                env = StopAllCars(env)
                # env = StopAndGo(env)
            if speed_mode:
                env = SpeedMode(env)
            if train:
                env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False, False, False) # train on default
        test_env = create_env(False, False, False, False) # evaluate on default
        test_env_modif = create_env(False, False, True, False) # evaluate on mod
        renderer = FreewayRenderer()

    elif config.get("ENV_NAME", None) == "Seaquest":
        from symbolic_options.reward_functions.seaquest import collect_divers_reward, fight_enemies_reward, upward_reward, shaped_reward, env_reward, total_rescued, total_collected, total_shot, total_surface_without_dying 
        from symbolic_options.reward_functions.seaquest import learned_meta_policy, llm_meta_policy, combined_meta_policy 
        from symbolic_options.reward_functions.seaquest import shoot_default_policy as conditional_meta_policy #conditional_meta_policy
        from jaxatari.wrappers import MultiRewardLogWrapper
        from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer
        from jaxatari.games.mods.seaquest_mods import DisableEnemiesWrapper
        # NOTE: the order of the rewards needs to align with the LLM-based meta-policy
        # NOTE: if conditional or combined provide idle_reward (not necessary for llm and learned)
        # this makes sure that there is always a fallback if no rule evaluates to true
        reward_funcs = [fight_enemies_reward, collect_divers_reward, upward_reward, total_rescued, total_collected, total_shot, total_surface_without_dying]
        if config.get("NO_REWARDS", False):
            reward_funcs = [env_reward, env_reward, env_reward] #use env_reward for all options
        # Shaped reward is reward function for meta-policy (not necessary, if meta-policy does not learn) 
        if config.get("META_SHAPED_REWARD", False):
            reward_funcs.append(shaped_reward)

        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        def create_env(train: bool = False, no_enemies: bool = False):
            env = JaxSeaquest(reward_funcs=reward_funcs)
            if no_enemies:
                env = DisableEnemiesWrapper(env)
            if train:
                env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False)
        test_env = create_env(False, False)
        test_env_modif = create_env(False, True)
        renderer = SeaquestRenderer()
    elif config.get("ENV_NAME", None) == "Kangaroo":
        from symbolic_options.reward_functions.kangaroo import navigate_reward, handle_enemies_reward, collect_fruits_reward, env_reward, reached_platform_level, enemies_killed, fruits_collected 
        # from symbolic_options.reward_functions.kangaroo import obstacle_avoidance_reward, vertical_navigation_reward, fruit_collection_reward
        from symbolic_options.reward_functions.kangaroo import llm_meta_policy, learned_meta_policy, combined_meta_policy, conditional_meta_policy
        from jaxatari.wrappers import MultiRewardLogWrapper#, MultiRewardWrapper
        from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooRenderer
        from jaxatari.games.mods.kangaroo_mods import DisableThreadsWrapper 

        reward_funcs = [navigate_reward, handle_enemies_reward, collect_fruits_reward, reached_platform_level, enemies_killed, fruits_collected] 
        # reward_funcs = [vertical_navigation_reward, obstacle_avoidance_reward, fruit_collection_reward, reached_platform_level, enemies_killed, fruits_collected]
        if config.get("NO_REWARDS", False):
            reward_funcs = [env_reward, env_reward, env_reward] #use env_reward for all options

        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        def create_env(train=False, no_enemies: bool = False):
            env = JaxKangaroo(reward_funcs=reward_funcs)
            if no_enemies:
                env = DisableThreadsWrapper(env)
            if train:
                env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life, first_fire=False)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False, first_fire=False)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False)
        test_env = create_env(False, False)
        test_env_modif = create_env(False, True)
        renderer = KangarooRenderer()

    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
    jax.random.split(rng, n_envs)#, env_params
    )
    vmap_step = lambda env_state, action: jax.vmap(
        env.step#, in_axes=(0, 0, None)
    )(env_state, action)#, env_params)

    vmap_eval_reset = lambda n_envs: lambda rng: jax.vmap(test_env.reset)(
    jax.random.split(rng, n_envs)
    )
    vmap_eval_step = lambda env_state, action: jax.vmap(
        test_env.step
    )(env_state, action)

    vmap_modif_reset = lambda n_envs: lambda rng: jax.vmap(test_env_modif.reset)(
        jax.random.split(rng, n_envs)
    )
    vmap_modif_step = lambda env_state, action: jax.vmap(
        test_env_modif.step
    )(env_state, action)

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
            env.action_space().n, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        # init_x = jnp.zeros(config["OBS_SHAPE"])
        init_x = jnp.zeros(env.observation_space().shape)
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
        obsv, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        def get_test_metrics(train_state, modif, rng):
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = jax.lax.cond(
                modif,
                lambda _: vmap_modif_reset(config["TEST_NUM_ENVS"])(_rng),
                lambda _: vmap_eval_reset(config["TEST_NUM_ENVS"])(_rng),
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
                    lambda _: vmap_modif_step(env_state, action),
                    lambda _: vmap_eval_step(env_state, action),
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
                obsv, env_state, reward, done, info = vmap_step(env_state, action)
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