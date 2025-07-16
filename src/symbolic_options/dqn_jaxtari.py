"""
PureJaxRL version of CleanRL's DQN: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py
"""
import time
import jax
import jax.numpy as jnp

import chex
import flax
import wandb
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import flashbax as fbx
from rtpt import RTPT

from symbolic_options.utils.video_recorder import video_callback


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class TimeStep:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array


class CustomTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int

rtpt = None
def rtpt_callback():
    global rtpt
    rtpt.step()


from jaxatari.wrappers import MultiRewardLogWrapper
from jaxatari.wrappers import ObjectCentricWrapper, FlattenObservationWrapper, AtariWrapper

def make_train(config):
    global rtpt

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_ENVS"]
    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=config["ALG_NAME"], max_iterations=config["NUM_UPDATES"])
    rtpt.start()

    if config["ENV_NAME"] == "Seaquest":
        from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer
        from symbolic_options.reward_functions.seaquest import collect_divers_reward, fight_enemies_reward, upward_reward, shaped_reward, env_reward, total_rescued, total_collected, total_shot, total_surface_without_dying 
        from jaxatari.games.mods.seaquest_mods import DisableEnemiesWrapper
        # this makes sure that there is always a fallback if no rule evaluates to true
        reward_funcs = [fight_enemies_reward, collect_divers_reward, upward_reward, total_rescued, total_collected, total_shot, total_surface_without_dying]

        def create_env(train: bool = False, no_enemies: bool = False):
            env = JaxSeaquest(reward_funcs=reward_funcs)
            if no_enemies:
                env = DisableEnemiesWrapper(env)
            if train:
                env = AtariWrapper(env, sticky_actions=True, episodic_life=True)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False)
        eval_env = create_env(False, False)
        modif_env = create_env(False, True)
        renderer = SeaquestRenderer()

    elif config["ENV_NAME"] == "Kangaroo":
        from symbolic_options.reward_functions.kangaroo import navigate_reward, handle_enemies_reward, collect_fruits_reward, env_reward, reached_platform_level, enemies_killed, fruits_collected
        from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooRenderer
        from jaxatari.games.mods.kangaroo_mods import DisableThreadsWrapper

        reward_funcs = [navigate_reward, handle_enemies_reward, collect_fruits_reward, reached_platform_level, enemies_killed, fruits_collected] 

        def create_env(train=False, no_enemies: bool = False):
            env = JaxKangaroo(reward_funcs=reward_funcs)
            if no_enemies:
                env = DisableThreadsWrapper(env)
            if train:
                env = AtariWrapper(env, sticky_actions=True, episodic_life=True)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False)
        eval_env = create_env(False, False)
        modif_env = create_env(False, True)
        renderer = KangarooRenderer()


    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
    jax.random.split(rng, n_envs)#, env_params
    )
    vmap_step = lambda env_state, action: jax.vmap(
        env.step#, in_axes=(0, 0, None)
    )(env_state, action)#, env_params)

    vmap_eval_reset = lambda n_envs: lambda rng: jax.vmap(eval_env.reset)(
    jax.random.split(rng, n_envs)
    )
    vmap_eval_step = lambda env_state, action: jax.vmap(
        eval_env.step
    )(env_state, action)

    vmap_modif_reset = lambda n_envs: lambda rng: jax.vmap(modif_env.reset)(
        jax.random.split(rng, n_envs)
    )
    vmap_modif_step = lambda env_state, action: jax.vmap(
        modif_env.step
    )(env_state, action)

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_sequences=False,
            add_batch_size=config["NUM_ENVS"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = env.action_space().sample(rng)
        _, _env_state = env.reset(rng)
        _obs, _, _reward, _done, _ = env.step(_env_state, _action)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state = buffer.init(_timestep)

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(action_dim=env.action_space().n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        network_params = network.init(_rng, init_x)

        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac

        lr = linear_schedule if config.get("LR_LINEAR_DECAY", False) else config["LR"]
        tx = optax.adam(learning_rate=lr)

        train_state = CustomTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            target_network_params=jax.tree.map(lambda x: jnp.copy(x), network_params),
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        # epsilon-greedy exploration
        def eps_greedy_exploration(rng, q_vals, t):
            rng_a, rng_e = jax.random.split(
                rng, 2
            )  # a key for sampling random actions and one for picking
            eps = jnp.clip(  # get epsilon
                (
                    (config["EPSILON_FINISH"] - config["EPSILON_START"])
                    / config["EPSILON_ANNEAL_TIME"]
                )
                * t
                + config["EPSILON_START"],
                config["EPSILON_FINISH"],
            )
            greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
            chosed_actions = jnp.where(
                jax.random.uniform(rng_e, greedy_actions.shape)
                < eps,  # pick the actions that should be random
                jax.random.randint(
                    rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
                ),  # sample random actions,
                greedy_actions,
            )
            return chosed_actions

        def get_test_metrics(train_state, modif, rng):
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = jax.lax.cond(
                modif,
                lambda _: vmap_eval_reset(config["TEST_NUM_ENVS"])(_rng),
                lambda _: vmap_modif_reset(config["TEST_NUM_ENVS"])(_rng),
                operand=None
            )

            def _test_step(runner_state, unused):
                env_state, last_obs, rng = runner_state
                rng, rng_a = jax.random.split(rng)
                q_vals = network.apply(train_state.params, last_obs)
                action = jnp.argmax(q_vals, axis=-1) # no exploration during testing
                obs, env_state, reward, done, info = jax.lax.cond(
                    modif,
                    lambda _: vmap_eval_step(env_state, action),
                    lambda _: vmap_modif_step(env_state, action),
                    operand=None
                )
                runner_state = (env_state, obs, rng)
                # info.pop("all_rewards")
                env_state_vid = jax.tree.map(lambda x: x[0], env_state)
                return runner_state, (info, env_state_vid, done[0])

            rng, _rng = jax.random.split(rng)
            runner_state = (env_state, init_obs, _rng)
            runner_state, output = jax.lax.scan(
                _test_step, runner_state, None, config["TEST_NUM_STEPS"]
            )
            metrics, states, dones = output

            # Aggregate metrics
            test_metrics = {}
            pre_str = "modif/" if modif else "test/"
            for k, v in metrics.items():
            # for k, v in done_infos.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        test_metrics[f"{pre_str}{k}_{sub_k}"] = sub_v.mean()
                else:
                    test_metrics[f"{pre_str}{k}"] = v.mean()

            # Record video
            if config["RECORD_VIDEO"]:
                jax.lax.cond(
                    train_state.n_updates > 0,
                    lambda _: jax.debug.callback(video_callback, states, None, None, dones, train_state.n_updates, renderer, modif=modif),
                    lambda _: None,
                    operand=None,
                )

            return test_metrics


        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(train_state.params, last_obs)
            action = eps_greedy_exploration(
                rng_a, q_vals, train_state.timesteps
            )  # explore with epsilon greedy_exploration
            obs, env_state, reward, done, info = vmap_step(env_state, action)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            # NETWORKS UPDATE
            def _learn_phase(train_state, rng):

                learn_batch = buffer.sample(buffer_state, rng).experience

                q_next_target = network.apply(
                    train_state.target_network_params, learn_batch.second.obs
                )  # (batch_size, num_actions)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                target = (
                    learn_batch.first.reward
                    + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
                )

                def _loss_fn(params):
                    q_vals = network.apply(
                        params, learn_batch.first.obs
                    )  # (batch_size, num_actions)
                    chosen_action_qvals = jnp.take_along_axis(
                        q_vals,
                        jnp.expand_dims(learn_batch.first.action, axis=-1),
                        axis=-1,
                    ).squeeze(axis=-1)
                    return jnp.mean((chosen_action_qvals - target) ** 2)

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                return train_state, loss

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                (buffer.can_sample(buffer_state))
                & (  # enough experience in buffer
                    train_state.timesteps > config["LEARNING_STARTS"]
                )
                & (  # pure exploration phase ended
                    train_state.timesteps % config["TRAINING_INTERVAL"] == 0
                )  # training interval
            )
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: _learn_phase(train_state, rng),
                lambda train_state, rng: (train_state, jnp.array(0.0)),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.timesteps % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            metrics = {
                "timesteps": train_state.timesteps,
                "updates": train_state.n_updates,
                "loss": loss.mean(),
                # "returns": info["returned_episode_returns"].mean(),
            }

            metrics.update({k: v.mean() for k, v in info.items()})

            # Get test metrics
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                train_state.timesteps % config["TEST_INTERVAL"] == 0,
                lambda ts, r: get_test_metrics(ts, False, r),
                lambda ts, r: {k: jnp.array(jnp.nan) for k in [f"test/{mk}" for mk in info.keys()]}, # Dummy zeros
                train_state,
                _rng
            )
            metrics.update(test_metrics)

            modif_metrics = jax.lax.cond(
                train_state.timesteps % config["TEST_INTERVAL"] == 0,
                lambda ts, r: get_test_metrics(ts, True, r),
                lambda ts, r: {k: jnp.array(jnp.nan) for k in [f"modif/{mk}" for mk in info.keys()]}, # Dummy zeros
                train_state,
                _rng
            )
            metrics.update(modif_metrics)


            # report on wandb if required
            if config.get("WANDB_MODE", "disabled") == "online":

                def callback(metrics):
                    # if metrics["timesteps"] % 100 == 0:
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            jax.debug.callback(rtpt_callback)

            runner_state = (train_state, buffer_state, env_state, obs, rng)

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, env_state, init_obs, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def main():
    # clean_rl HP's
    config = {
        "NUM_ENVS": 1,
        "BUFFER_SIZE": 100_000,
        "BUFFER_BATCH_SIZE": 32,
        # "BUFFER_BATCH_SIZE": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.01,
        "EPSILON_ANNEAL_TIME": 25e4, #in steps (not frames) 0.1 (==250_000steps)
        "TARGET_UPDATE_INTERVAL": 1000,
        "LR": 1e-4,
        "LEARNING_STARTS": 80_000,
        "TRAINING_INTERVAL": 4,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "TAU": 1.0,
        "ENV_NAME": "Seaquest",
        "SEED": 0,
        "NUM_SEEDS": 1,
        "WANDB_MODE": "online",  # set to online to activate wandb
        "ENTITY": "",
        "PROJECT": "",
        "NAME_INITIALS": "RE",
        "ALG_NAME": "DQN",
        "TEST_INTERVAL": 100_000,
        "TEST_NUM_ENVS": 10,
        "TEST_NUM_STEPS": 10_000,
        "RECORD_VIDEO": True,  # Whether to record video or not
    }

    # Nature DQN config (but with adam 1e-4 instead of RMSProp 2.5e-4)
    # config = {
    #     "NUM_ENVS": 1,
    #     "BUFFER_SIZE": 100_000,
    #     "BUFFER_BATCH_SIZE": 32,
    #     # "BUFFER_BATCH_SIZE": 128,
    #     "TOTAL_TIMESTEPS": 5e7,
    #     "EPSILON_START": 1.0,
    #     "EPSILON_FINISH": 0.1,
    #     "EPSILON_ANNEAL_TIME": 25e4,# in steps (not frames) 0.02 (==250000steps) rather than 0.1
    #     "TARGET_UPDATE_INTERVAL": 10_000,
    #     "LR": 1e-4,
    #     "LEARNING_STARTS": 50_000,
    #     "TRAINING_INTERVAL": 4,
    #     "LR_LINEAR_DECAY": False,
    #     "GAMMA": 0.99,
    #     "TAU": 1.0,
    #     "ENV_NAME": "Seaquest", # "Kangaroo" or "Seaquest"
    #     "SEED": 0,
    #     "NUM_SEEDS": 1,
    #     "WANDB_MODE": "online",  # set to online to activate wandb
    #     "ENTITY": "",
    #     "PROJECT": "",
    #     "NAME_INITIALS": "RE",
    #     "ALG_NAME": "DQN",
    #     "TEST_INTERVAL": 100_000,
    #     "TEST_NUM_ENVS": 10,
    #     "TEST_NUM_STEPS": 10_000,
    #     "RECORD_VIDEO": True,  # Whether to record video or not
    # }

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["DQN", config["ENV_NAME"].upper(), f"jax_{jax.__version__}"],
        name=f'purejaxrl_dqn_{config["ENV_NAME"]}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    print("Compiling...")
    start = time.time()
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    train_vjit.lower(rngs).compile()
    print(f"Compilation took {time.time() - start:.2f} seconds")
    outs = jax.block_until_ready(train_vjit(rngs))


if __name__ == "__main__":
    main()