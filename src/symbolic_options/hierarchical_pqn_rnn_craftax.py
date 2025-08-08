import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any
from rtpt import RTPT

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

import wandb

from symbolic_options.purejaxql.batch_renorm import BatchRenorm
from symbolic_options.utils.video_recorder import video_callback


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state[0].shape[-1]

        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree.map(
            lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
            init_rnn_state,
            rnn_state,
        )

        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class RNNQNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input in any case for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        new_hidden = []
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)

        q_vals = nn.Dense(self.action_dim)(x)

        return new_hidden, q_vals

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]


@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    agent: chex.Array
    rewards: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    q_val: chex.Array
    meta_q_val: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

rtpt = None
def rtpt_callback():
    global rtpt
    rtpt.step()

def make_train(config, env, test_env, test_env_modif, env_params, meta_policy, meta_policy_llm, renderer):
    global rtpt

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

    rtpt = RTPT(name_initials=config["NAME_INITIALS"], experiment_name=config["ALG_NAME"], max_iterations=config["NUM_UPDATES"])
    rtpt.start()

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions


    def train(rng, params, batch_stats):

        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        eps_meta_scheduler = optax.linear_schedule(
            config["META_EPS_START"],
            config["META_EPS_FINISH"],
            config["META_EPS_DECAY"] * (config["NUM_UPDATES_DECAY"] - config["PRETRAIN_LEN"]),
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        lr_scheduler_meta = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"] - config["PRETRAIN_LEN"],
        )
        lr_meta = lr_scheduler_meta if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = RNNQNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng, params, batch_stats, network, lr):
            init_x = (
                jnp.zeros(
                    (1, 1, *env.observation_space(env_params).shape)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )  # (obs, dones, last_actions)
            init_hs = network.initialize_carry(1)  # (batch_size, hidden_dim)
            network_variables = network.init(rng, init_hs, *init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"] if params is None else params,
                batch_stats=network_variables["batch_stats"] if batch_stats is None else batch_stats,
                tx=tx,
            )
            return train_state

        num_agents = config.get("NUM_AGENTS", len(env.reward_funcs))
        if num_agents == 0:
            num_agents = 1

        # create multiple agents
        rng_keys = jax.random.split(rng, num_agents)
        train_states: CustomTrainState = jax.vmap(create_agent, in_axes=(0, 0, 0, None, None))(rng_keys, params, batch_stats, network, lr)

        meta_network = RNNQNetwork(
            action_dim=num_agents,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )
        meta_train_state = create_agent(rng, None, None, meta_network, lr_meta)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_states, memory_transitions, meta_train_state, expl_state, test_metrics, test_metrics_modif, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

                def compute_q_values(train_state, hs):
                    new_hs, q_vals = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        hs,
                        _obs,
                        _done,
                        _last_action,
                        train=False,
                    )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                    q_vals = q_vals.squeeze(axis=0)
                    return new_hs, q_vals

                new_hs, all_q_vals = jax.vmap(compute_q_values)(train_states, hs)

                def compute_actions(q_vals, train_state):
                    _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                    eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                    return new_action

                all_actions = jax.vmap(compute_actions)(all_q_vals, train_states)

                meta_policy_mode = config.get("META_POLICY", "llm")
                llm_pretrain = config.get("LLM_PRETRAIN", False)
                random_pretrain = config.get("RANDOM_PRETRAIN", False)

                def handle_conditional():
                    active_agent_q_vals = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    agent_probs = jax.nn.softmax(active_agent_q_vals, axis=-1)
                    max_q_vals = jnp.transpose(jnp.max(all_q_vals, axis=-1))
                    return agent_probs * max_q_vals

                def handle_llm_pretrain():
                    llm_q = meta_policy_llm(meta_network, meta_train_state, last_obs, env_state)
                    both_q = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    return jax.lax.cond(
                        train_states.n_updates[0] < config["PRETRAIN_LEN"],
                        lambda _: llm_q,
                        lambda _: both_q,
                        operand=None
                    )

                def handle_random_pretrain():
                    both_q = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    random_q = jax.random.randint(
                        rng_a, shape=both_q.shape, minval=0, maxval=both_q.shape[-1]
                    ).astype(jnp.float32)
                    return jax.lax.cond(
                        train_states.n_updates[0] < config["PRETRAIN_LEN"],
                        lambda _: random_q,
                        lambda _: both_q,
                        operand=None
                    )

                def handle_default():
                    return meta_policy(meta_network, meta_train_state, last_obs, env_state)

                combined_q = jax.lax.switch(
                    jnp.array([
                        llm_pretrain,
                        random_pretrain,
                        meta_policy_mode == "conditional",
                        True
                    ], dtype=jnp.bool_).argmax(),
                    [handle_llm_pretrain, handle_random_pretrain, handle_conditional, handle_default],
                )

                use_greedy = config.get("META_GREEDY", True)

                def select_greedy(_):
                    return jnp.argmax(combined_q, axis=-1)

                def select_exploratory(_):
                    rng_split = jax.random.split(rng_s, config["NUM_ENVS"])
                    eps = jnp.full((config["NUM_ENVS"],), eps_meta_scheduler(meta_train_state.n_updates))
                    return jax.vmap(eps_greedy_exploration)(rng_split, combined_q, eps)

                active_agent = jax.lax.cond(
                    use_greedy,
                    select_greedy,
                    select_exploratory,
                    operand=None
                )

                new_action = all_actions[active_agent, jnp.arange(config["NUM_ENVS"])]

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                rewards = info.pop("all_rewards")
                rewards = jnp.concatenate((rewards, reward[:, None]), axis=1)

                transition = Transition(
                    # last_hs=hs,
                    last_hs=jax.tree.map(lambda x: jnp.transpose(x, (1, 0, 2)), hs),  # (num_envs, num_agents, hidden_size)
                    obs=last_obs,
                    action=new_action,
                    agent=active_agent,
                    rewards=config.get("REW_SCALE", 1) * rewards,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    # q_val=all_q_vals,
                    q_val=jax.tree.map(lambda x: jnp.transpose(x, (1, 0, 2)), all_q_vals),  # (num_envs, num_agents, num_actions)
                    meta_q_val=combined_q
                )
                # new_hs = all_hs[active_agent, jnp.arange(config["NUM_ENVS"])]
                return (new_hs, new_obs, new_done, new_action, new_env_state, rng), (
                    transition,
                    info,
                )

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            # insert the transitions into the memory
            memory_transitions = jax.tree.map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
                memory_transitions,
                transitions,
            )

            def _update_agent(train_state, state_idx, rng, network):
                train_state = train_state.replace(
                    timesteps=train_state.timesteps
                    + config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                            # NETWORKS UPDATE
                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch):

                        # minibatch shape: num_steps, batch_size, ...
                        # with batch_size = num_envs/num_minibatches

                        train_state, rng = carry
                        #TODO: Since this is done, we should probably directly only store last_hs[0] and save memory
                        hs = jax.tree.map(lambda x: x[0], minibatch.last_hs)  # hs of oldest step (batch_size, hidden_size)
                        # Also select the hs of the active agent
                        hs = jax.tree.map(lambda x: x[:, state_idx], hs)
                        agent_in = (
                            minibatch.obs,
                            minibatch.last_done,
                            minibatch.last_action,
                        )

                        def _compute_targets(last_q, q_vals, meta_q_vals, rewards, done):
                            def _get_target(lambda_returns_and_next_q, rew_q_done):
                                rewards, q, meta_q, done = rew_q_done
                                lambda_returns, next_q = lambda_returns_and_next_q
                                target_bootstrap = (
                                    # rewards[..., state_idx] + config["GAMMA"] * (1 - done) * next_q
                                    rewards + config["GAMMA"] * (1 - done) * next_q
                                )
                                delta = lambda_returns - next_q
                                lambda_returns = (
                                    target_bootstrap
                                    + config["GAMMA"] * config["LAMBDA"] * delta
                                )
                                # lambda_returns = (1 - done) * lambda_returns + done * rewards[..., state_idx]
                                lambda_returns = (1 - done) * lambda_returns + done * rewards
                                # next_q = jnp.max(q, axis=-1) # this would be q of best action
                                # instead, we use the q of the currently active option (or meta)
                                next_q = jax.lax.cond(
                                    state_idx == num_agents,
                                    lambda _: jnp.max(meta_q, axis=-1),
                                    lambda _: jnp.max(q, axis=-1),
                                    operand=None,
                                )
                                return (lambda_returns, next_q), lambda_returns

                            lambda_returns = (
                                # rewards[-1, :, state_idx] + config["GAMMA"] * (1 - done[-1]) * last_q
                                rewards[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                            )
                            _, targets = jax.lax.scan(
                                _get_target,
                                (lambda_returns, last_q),
                                jax.tree.map(lambda x: x[:-1], (rewards, q_vals, meta_q_vals, done)),
                                reverse=True,
                            )
                            targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                            return targets

                        def _loss_fn(params):
                            (_, q_vals), updates = partial(
                                network.apply, train=True, mutable=["batch_stats"]
                            )(
                                {"params": params, "batch_stats": train_state.batch_stats},
                                hs,
                                *agent_in,
                            )  # (num_steps, batch_size, num_actions)

                            # lambda returns are computed using NUM_STEPS as the horizon, and optimizing from t=0 to NUM_STEPS-1
                            target_q_vals = jax.lax.stop_gradient(q_vals)
                            last_q = target_q_vals[-1].max(axis=-1) # would be max_q_vals of prev last step
                            target = _compute_targets(
                                last_q,  # q_vals at t=NUM_STEPS-1
                                target_q_vals[:-1], # is already only q_vals of active agent
                                minibatch.meta_q_val[:-1],
                                # minibatch.rewards[:-1],
                                minibatch.rewards[:-1, ..., state_idx],  # rewards of active agent
                                minibatch.done[:-1],
                            ).reshape(
                                -1
                            )  # (num_steps-1*batch_size,)

                            # chosen_action_qvals = jnp.take_along_axis(
                            #     q_vals,
                            #     jnp.expand_dims(minibatch.action, axis=-1),
                            #     axis=-1,
                            # ).squeeze(
                            #     axis=-1
                            # )  # (num_steps, num_agents, batch_size,)
                            # chosen_action_qvals = chosen_action_qvals[:-1].reshape(
                            #     -1
                            # )  # (num_steps-1*batch_size,)

                            chosen_action_qvals = jax.lax.cond(
                                state_idx == num_agents,
                                lambda _: jnp.take_along_axis(
                                    q_vals,
                                    jnp.expand_dims(minibatch.agent, axis=-1),
                                    axis=-1,
                                ).squeeze(axis=-1),
                                lambda _: jnp.take_along_axis(
                                    q_vals,
                                    jnp.expand_dims(minibatch.action, axis=-1),
                                    axis=-1,
                                ).squeeze(axis=-1),
                                operand=None,
                            )
                            chosen_action_qvals = chosen_action_qvals[:-1].reshape(
                                -1
                            )

                            loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                            return loss, (updates, chosen_action_qvals)

                        (loss, (updates, qvals)), grads = jax.value_and_grad(
                            _loss_fn, has_aux=True
                        )(train_state.params)
                        train_state = train_state.apply_gradients(grads=grads)
                        train_state = train_state.replace(
                            grad_steps=train_state.grad_steps + 1,
                            batch_stats=updates["batch_stats"],
                        )
                        return (train_state, rng), (loss, qvals)

                    def preprocess_transition(x, rng):
                        # x: (num_steps, num_envs, ...)
                        x = jax.random.permutation(
                            rng, x, axis=1
                        )  # shuffle the transitions
                        x = x.reshape(
                            x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
                        )  # num_steps, minibatches, batch_size/num_minbatches,
                        x = jnp.swapaxes(
                            x, 0, 1
                        )  # (minibatches, num_steps, batch_size/num_minbatches, ...)
                        return x

                    rng, _rng = jax.random.split(rng)
                    minibatches = jax.tree.map(
                        lambda x: preprocess_transition(x, _rng),
                        memory_transitions,
                    )  # num_minibatches, num_steps+memory_window, batch_size/num_minbatches, ...

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), (loss, qvals) = jax.lax.scan(
                        _learn_phase, (train_state, rng), minibatches
                    )

                    return (train_state, rng), (loss, qvals)

                # def _learn_epoch(carry, _):
                #     train_state, rng = carry

                #     def _learn_phase(carry, minibatch_and_target):
                #         train_state, rng = carry
                #         minibatch, target = minibatch_and_target

                #         def _loss_fn(params):
                #             hs = jax.tree.map(lambda x: x[0], minibatch.last_hs)
                #             agent_in = (
                #                 minibatch.obs,
                #                 minibatch.last_done,
                #                 minibatch.last_action,
                #             )
                #             (_, q_vals), updates = partial(
                #                 network.apply, train=True, mutable=["batch_stats"]
                #             )(
                #                 {"params": params, "batch_stats": train_state.batch_stats},
                #                 hs,
                #                 *agent_in,
                #             )

                #             chosen_action_qvals = jnp.take_along_axis(
                #                 q_vals,
                #                 jnp.expand_dims(minibatch.action, axis=-1),
                #                 axis=-1,
                #             ).squeeze(axis=-1)

                #             loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                #             return loss, (updates, chosen_action_qvals)

                #         (loss, (updates, qvals)), grads = jax.value_and_grad(
                #             _loss_fn, has_aux=True
                #         )(train_state.params)
                #         train_state = train_state.apply_gradients(grads=grads)
                #         train_state = train_state.replace(
                #             grad_steps=train_state.grad_steps + 1,
                #             batch_stats=updates["batch_stats"],
                #         )
                #         return (train_state, rng), (loss, qvals)

                #     def preprocess_transition(x, rng):
                #         x = x.reshape(
                #             -1, *x.shape[2:]
                #         )
                #         x = jax.random.permutation(rng, x)
                #         x = x.reshape(
                #             config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                #         )
                #         return x

                #     rng, _rng = jax.random.split(rng)
                #     minibatches = jax.tree.map(
                #         lambda x: preprocess_transition(x, _rng), transitions
                #     )
                #     targets = jax.tree.map(
                #         lambda x: preprocess_transition(x, _rng), lambda_targets
                #     )

                #     rng, _rng = jax.random.split(rng)
                #     (train_state, rng), (loss, qvals) = jax.lax.scan(
                #         _learn_phase, (train_state, rng), (minibatches, targets)
                #     )

                #     return (train_state, rng), (loss, qvals)

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
                )

                train_state = train_state.replace(n_updates=train_state.n_updates + 1)
                eps = jax.lax.cond(
                    state_idx == num_agents,
                    lambda _: eps_meta_scheduler(train_state.n_updates),
                    lambda _: eps_scheduler(train_state.n_updates),
                    operand=None,
                )
                metrics = {
                    "env_step": train_state.timesteps,
                    "update_steps": train_state.n_updates,
                    "grad_steps": train_state.grad_steps,
                    "td_loss": loss.mean(),
                    "qvals": qvals.mean(),
                    "eps": eps,
                }
                return metrics, train_state

            rngs = jax.random.split(rng, num_agents)
            metrics, train_states = jax.vmap(_update_agent, in_axes=(0, 0, 0, None))(train_states, jnp.arange(num_agents), rngs, network)
            metrics = {f"{k}_{i}": v[i] for k, v in metrics.items() for i in range(v.shape[0])}

            meta_policy_string = config.get("META_POLICY", "llm")
            meta_policy_can_learn = (meta_policy_string == "learned") or (meta_policy_string == "combined")

            meta_reward_idx = -1
            def do_update(_):
                return _update_agent(meta_train_state, meta_reward_idx, rng, meta_network)

            def skip_update(_):
                new_train_state = meta_train_state.replace(
                    timesteps=meta_train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                metrics_meta, _ = _update_agent(new_train_state, meta_reward_idx, rng, meta_network)
                return metrics_meta, new_train_state

            is_done_pretraining = train_states.n_updates[0] > config["PRETRAIN_LEN"]
            is_not_pretraining = not (config.get("LLM_PRETRAIN", False) or config.get("RANDOM_PRETRAIN", False))
            should_update = jnp.logical_or(is_done_pretraining, is_not_pretraining)
            should_update = jnp.logical_or(should_update, meta_policy_can_learn)

            meta_metrics, meta_train_state = jax.lax.cond(
                should_update,
                do_update,
                skip_update,
                operand=None,
            )
            metrics.update({f"meta_{k}": v for k, v in meta_metrics.items()})

            metrics.update({k: jnp.nanmean(v) for k, v in infos.items()}),

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_states.n_updates[0] % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_test_metrics(train_states, meta_train_state, False, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

                if config.get("TEST_MODIFS", False):
                    rng, _rng = jax.random.split(rng)
                    test_metrics_modif = jax.lax.cond(
                        train_states.n_updates[0] % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                        lambda _: get_test_metrics(train_states, meta_train_state, True, _rng),
                        lambda _: test_metrics_modif,
                        operand=None,
                    )
                    metrics.update({f"test_modif/{k}": v for k, v in test_metrics_modif.items()})

            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }

            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_rng)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps_0"])
                jax.debug.callback(callback, metrics, original_rng)
            jax.debug.callback(rtpt_callback)

            runner_state = (
                train_states,
                memory_transitions,
                meta_train_state,
                tuple(expl_state),
                test_metrics,
                test_metrics_modif,
                rng,
            )

            return runner_state, metrics

        def get_test_metrics(train_states, meta_train_state, modif, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, prev_rewards, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                def compute_actions(train_state):
                    q_vals = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        last_obs,
                        train=False,
                    )
                    _rngs = jax.random.split(rng_a, config["TEST_NUM_ENVS"])
                    eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                    return new_action, q_vals

                all_actions, all_q_vals = jax.vmap(compute_actions)(train_states)

                meta_policy_mode = config.get("META_POLICY", "llm")
                llm_pretrain = config.get("LLM_PRETRAIN", False)
                random_pretrain = config.get("RANDOM_PRETRAIN", False)

                def handle_conditional():
                    active_agent_q_vals = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    agent_probs = jax.nn.softmax(active_agent_q_vals, axis=-1)
                    max_q_vals = jnp.transpose(jnp.max(all_q_vals, axis=-1))
                    return agent_probs * max_q_vals

                def handle_llm_pretrain():
                    llm_q = meta_policy_llm(meta_network, meta_train_state, last_obs, env_state)
                    both_q = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    return jax.lax.cond(
                        train_states.n_updates[0] < config["PRETRAIN_LEN"],
                        lambda _: llm_q,
                        lambda _: both_q,
                        operand=None
                    )

                def handle_random_pretrain():
                    both_q = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    random_q = jax.random.randint(
                        rng_a, shape=both_q.shape, minval=0, maxval=both_q.shape[-1]
                    ).astype(jnp.float32)
                    return jax.lax.cond(
                        train_states.n_updates[0] < config["PRETRAIN_LEN"],
                        lambda _: random_q,
                        lambda _: both_q,
                        operand=None
                    )

                def handle_default():
                    return meta_policy(meta_network, meta_train_state, last_obs, env_state)

                combined_q = jax.lax.switch(
                    jnp.array([
                        llm_pretrain,
                        random_pretrain,
                        meta_policy_mode == "conditional",
                        True
                    ], dtype=jnp.bool_).argmax(),
                    [handle_llm_pretrain, handle_random_pretrain, handle_conditional, handle_default],
                )

                use_greedy = config.get("META_GREEDY", True)

                def select_greedy(_):
                    return jnp.argmax(combined_q, axis=-1)

                def select_exploratory(_):
                    rng_split = jax.random.split(rng_s, config["TEST_NUM_ENVS"])
                    eps = jnp.full((config["TEST_NUM_ENVS"],), eps_meta_scheduler(meta_train_state.n_updates))
                    return jax.vmap(eps_greedy_exploration)(rng_split, combined_q, eps)

                active_agent = jax.lax.cond(
                    use_greedy,
                    select_greedy,
                    select_exploratory,
                    operand=None
                )

                new_action = all_actions[active_agent, jnp.arange(config["TEST_NUM_ENVS"])]

                new_obs, new_env_state, reward, new_done, info = jax.lax.cond(
                    modif,
                    lambda _: test_env_modif.step(rng_s, env_state, new_action, env_params),
                    lambda _: test_env.step(rng_s, env_state, new_action, env_params),
                    operand=None
                )

                env_state_vid = jax.tree.map(lambda x: x[0], new_env_state)
                rewards = info.pop("all_rewards")[:, :num_agents]

                mask = jnp.arange(num_agents) == active_agent[:, None]
                active_rewards = jnp.where(mask, rewards, 0)

                new_prev_rewards = prev_rewards + active_rewards
                all_returns = new_prev_rewards
                all_done = jnp.repeat(new_done[:, None], num_agents, axis=1)
                all_returns = jnp.where(
                    all_done, all_returns, jnp.nan * jnp.ones_like(all_returns)
                )
                info["active_returns"] = all_returns[jnp.arange(config["TEST_NUM_ENVS"]), active_agent]
                for agent_idx in range(num_agents):
                    info[f"active_returns_{agent_idx}"] = all_returns[:, agent_idx]

                new_prev_rewards = jnp.where(
                    all_done, active_rewards, new_prev_rewards
                )

                return (new_env_state, new_obs, new_prev_rewards, rng), (info, env_state_vid, active_agent[0], combined_q[0], new_done[0])

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = jax.lax.cond(
                modif,
                lambda _: test_env_modif.reset(_rng, env_params),
                lambda _: test_env.reset(_rng, env_params),
                operand=None
            )

            init_rewards = jnp.zeros((config["TEST_NUM_ENVS"], num_agents))
            _, output = jax.lax.scan(
                _env_step, (env_state, init_obs, init_rewards, _rng), None, config["TEST_NUM_STEPS"]
            )
            infos, states, active_agents, combined_qs, dones = output

            if config.get("RECORD_VIDEO", False):
                jax.lax.cond(
                    train_states.n_updates[0] > 0,
                    lambda _: jax.debug.callback(video_callback, states, active_agents, combined_qs, dones, train_states.n_updates[0], renderer, modif=modif),
                    lambda _: None,
                    operand=None,
                )

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

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_states, meta_train_state, False, _rng)

        rng, _rng = jax.random.split(rng)
        test_metrics_modif = get_test_metrics(train_states, meta_train_state, True, _rng)

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        init_dones = jnp.zeros((config["NUM_ENVS"],), dtype=bool)
        init_action = jnp.zeros((config["NUM_ENVS"],), dtype=int)
        # init_hs = network.initialize_carry(config["NUM_ENVS"])
        # initialize the hidden states for each agent
        init_hs = jax.vmap(lambda x: network.initialize_carry(config["NUM_ENVS"]))(train_states)
        expl_state = (init_hs, obs, init_dones, init_action, env_state)

        # step randomly to have the initial memory window
        def _random_step(carry, _):
            hs, last_obs, last_done, last_action, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
            _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
            _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

            def compute_q_values(train_state, hs):
                new_hs, q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(axis=0)
                return (new_hs, q_vals)

            new_hs, all_q_vals = jax.vmap(compute_q_values)(train_states, hs)


            def compute_actions(q_vals, train_state):
                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], 1.0)  # random actions
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                return new_action

            all_actions = jax.vmap(compute_actions)(all_q_vals, train_states)
            active_agent = jax.random.randint(
                rng_a, shape=(config["NUM_ENVS"],), minval=0, maxval=num_agents-1
            )
            new_action = all_actions[active_agent, jnp.arange(config["NUM_ENVS"])]
            new_obs, new_env_state, reward, new_done, info = env.step(
                rng_s, env_state, new_action, env_params
            )
            rewards = info.pop("all_rewards")
            rewards = jnp.concatenate((rewards, reward[:, None]), axis=1)

            #TODO: Could adapt this to the actual strategyy
            meta_q_val = meta_policy(meta_network, meta_train_state, last_obs, env_state)

            transition = Transition(
                last_hs=jax.tree.map(lambda x: jnp.transpose(x, (1, 0, 2)), hs),  # (num_envs, num_agents, hidden_size)
                obs=last_obs,
                action=new_action,
                agent=active_agent,
                rewards=config.get("REW_SCALE", 1) * rewards,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                # q_val=all_q_vals,
                q_val=jax.tree.map(lambda x: jnp.transpose(x, (1, 0, 2)), all_q_vals),  # (num_envs, num_agents, num_actions)
                meta_q_val=meta_q_val
            )
            #TODO: remove
            # new_hs = jax.tree.map(
            #     lambda x: x[active_agent, jnp.arange(config["NUM_ENVS"])],
            #     all_hs,
            # )
            # new_hs = all_hs[active_agent, jnp.arange(config["NUM_ENVS"])]
            return (
                new_hs,
                new_obs,
                new_done,
                new_action,
                new_env_state,
                rng,
            ), transition

        rng, _rng = jax.random.split(rng)
        (*expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (*expl_state, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        # last hs-array in memory_transition has shape (128, 5, 1024, 512) -> (mem+steps, n_agents, n_envs, h_dim)
        # now it has shape (128, 1024, 5, 512) -> (mem+steps, n_agents, n_envs, h_dim)
        # But: for shuffling etc. to work, we need to have (mem+steps, n_envs, n_agents, h_dim)
        expl_state = tuple(expl_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, memory_transitions, meta_train_state, expl_state, test_metrics, test_metrics_modif, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train