import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
from rtpt import RTPT

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import wandb

from symbolic_options.purejaxql.batch_renorm import BatchRenorm
from symbolic_options.utils.video_recorder import video_callback

class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "batch_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            # dummy normalize input for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)

        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    agent: chex.Array
    rewards: chex.Array
    done: chex.Array
    next_obs: chex.Array
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

def make_train(config, env, test_env, test_env_modif, env_params, meta_policy, meta_policy_llm, renderer):# -> Callable[..., dict[str, Any]]:
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
        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng, params, batch_stats, network: QNetwork, lr):
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            network_variables = network.init(rng, init_x, train=False)
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
        # networks.append(meta_network)
        rng_keys = jax.random.split(rng, num_agents)
        train_states: CustomTrainState = jax.vmap(create_agent, in_axes=(0, 0, 0, None, None))(rng_keys, params, batch_stats, network, lr)

        meta_network = QNetwork(
            action_dim=num_agents,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )
        meta_train_state = create_agent(rng, None, None, meta_network, lr_meta)

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            
            train_states, meta_train_state, expl_state, test_metrics, test_metrics_modif, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                # use the meta-policy for stepping the environment 
                last_obs, env_state, rng = carry
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
                    # different eps for each env
                    _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                    eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                    return new_action, q_vals

                all_actions, all_q_vals = jax.vmap(compute_actions)(train_states)

                meta_policy_mode = config.get("META_POLICY", "llm")
                llm_pretrain = config.get("LLM_PRETRAIN", False)
                random_pretrain = config.get("RANDOM_PRETRAIN", False)

                # Combine Q-values depending on mode
                def handle_conditional():
                    active_agent_q_vals = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    agent_probs = jax.nn.softmax(active_agent_q_vals, axis=-1)  # (128, 3)
                    max_q_vals = jnp.transpose(jnp.max(all_q_vals, axis=-1))    # (128, 3)
                    return agent_probs * max_q_vals                              # (128, 3)

                def handle_llm_pretrain():
                    # llm pretrainining requires access to both, llm meta-policy and combined
                    llm_q= meta_policy_llm(meta_network, meta_train_state, last_obs, env_state)
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

                # Decide which Q-combination strategy to apply
                combined_q = jax.lax.switch(
                    jnp.array([
                        llm_pretrain,
                        random_pretrain,
                        meta_policy_mode == "conditional",
                        True # default if nothing else is true
                    ], dtype=jnp.bool_).argmax(),  # priority order
                    [handle_llm_pretrain, handle_random_pretrain, handle_conditional, handle_default],
                )

                # Now select active agent — greedy vs exploratory
                use_greedy = config.get("META_GREEDY", True)

                def select_greedy(_):
                    return jnp.argmax(combined_q, axis=-1)  # (128,)

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

                # select the q_vals and action of the active agent
                new_action = all_actions[active_agent, jnp.arange(config["NUM_ENVS"])] # (128,)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                # add reward to end -> (128,N_rews+1)
                rewards = info.pop("all_rewards") #(N_envs, N_rews)
                rewards = jnp.concatenate((rewards, reward[:, None]), axis=1)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    agent=active_agent,
                    rewards=config.get("REW_SCALE", 1)*rewards,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=all_q_vals,
                    meta_q_val=combined_q
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            # prev_rewards = jnp.zeros((config["NUM_ENVS"], num_agents))
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            def _update_agent(train_state, state_idx, rng, network): 
                # Update each network separately

                train_state = train_state.replace(
                    timesteps=train_state.timesteps
                    + config["NUM_STEPS"] * config["NUM_ENVS"]
                )  # update timesteps count

                last_q = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    transitions.next_obs[-1],
                    train=False,
                )
                last_q = last_q[..., state_idx]  # select the q_val of the active agent

                def _get_target(lambda_returns_and_next_q, transition):
                    lambda_returns, next_q = lambda_returns_and_next_q
                    target_bootstrap = (
                        transition.rewards[... , state_idx] + config["GAMMA"] * (1 - transition.done) * next_q
                    )
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (
                        1 - transition.done
                    ) * lambda_returns + transition.done * transition.rewards[... , state_idx]
                    next_q = jax.lax.cond(
                        state_idx == num_agents,
                        lambda _: jnp.max(transition.meta_q_val, axis=-1),
                        lambda _: jnp.max(transition.q_val[state_idx, jnp.arange(config["NUM_ENVS"]), :], axis=-1),
                        operand=None,
                    )
                    return (lambda_returns, next_q), lambda_returns

                last_q = last_q * (1 - transitions.done[-1])
                lambda_returns = transitions.rewards[-1, :, state_idx] + config["GAMMA"] * last_q
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree.map(lambda x: x[:-1], transitions),
                    reverse=True,
                )
                lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

                # NETWORKS UPDATE
                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_and_target):

                        train_state, rng = carry
                        minibatch, target = minibatch_and_target

                        def _loss_fn(params):
                            if config.get("Q_LAMBDA", False):
                                q_vals, updates = network.apply(
                                    {
                                        "params": params,
                                        "batch_stats": train_state.batch_stats,
                                    },
                                    minibatch.obs,
                                    train=True,
                                    mutable=["batch_stats"],
                                )
                            else:
                                # if not using q_lambda, re-pass the next_obs through the network to compute target
                                all_q_vals, updates = network.apply(
                                    {
                                        "params": params,
                                        "batch_stats": train_state.batch_stats,
                                    },
                                    jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                                    train=True,
                                    mutable=["batch_stats"],
                                )
                                q_vals, q_next = jnp.split(all_q_vals, 2)
                                q_next = jax.lax.stop_gradient(q_next)
                                q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
                                target = (
                                    minibatch.rewards[..., state_idx]
                                    + (1 - minibatch.done) * config["GAMMA"] * q_next
                                )

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
                        x = x.reshape(
                            -1, *x.shape[2:]
                        )  # num_steps*num_envs (batch_size), ...
                        x = jax.random.permutation(rng, x)  # shuffle the transitions
                        x = x.reshape(
                            config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                        )  # num_mini_updates, batch_size/num_mini_updates, ...
                        return x

                    rng, _rng = jax.random.split(rng)
                    minibatches = jax.tree.map(
                        lambda x: preprocess_transition(x, _rng), transitions
                    )  # num_actors*num_envs (batch_size), ...
                    targets = jax.tree.map(
                        lambda x: preprocess_transition(x, _rng), lambda_targets
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), (loss, qvals) = jax.lax.scan(
                        _learn_phase, (train_state, rng), (minibatches, targets)
                    )

                    return (train_state, rng), (loss, qvals)

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

            # end of _update_agent
            rngs = jax.random.split(rng, num_agents)
            metrics, train_states = jax.vmap(_update_agent, in_axes=(0, 0, 0, None))(train_states, jnp.arange(num_agents), rngs, network)
            # currently each key has a list of three values, make it s.t. we have key_0, key_1, key_2
            metrics = {f"{k}_{i}": v[i] for k, v in metrics.items() for i in range(v.shape[0])}
            
            meta_policy_string = config.get("META_POLICY", "llm")
            meta_policy_can_learn = (meta_policy_string == "learned") or (meta_policy_string == "combined")

            meta_reward_idx = -1 # see jaxtari version for explanation
            def do_update(_):
                return _update_agent(meta_train_state, meta_reward_idx, rng, meta_network)

            def skip_update(_):
                new_train_state = meta_train_state.replace(
                    timesteps=meta_train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                # run update step to get metrics, but don't change train_state
                metrics_meta, _ = _update_agent(new_train_state, meta_reward_idx, rng, meta_network)
                return metrics_meta, new_train_state

            # Only update if the meta-policy is "learned" or "combined"
            # and if we are not in pretraining or if we are done with pretraining
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

            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            metrics.update(done_infos)

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

            # remove achievement metrics if not logging them
            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":
                def callback(metrics, original_rng):
                    # log at intervals
                    if (
                        metrics["update_steps_0"] % config.get("WANDB_LOG_INTERVAL", 128) == 0
                    ):
                        if config.get("WANDB_LOG_ALL_SEEDS", False):
                            metrics.update(
                                {
                                    f"rng{int(original_rng)}/{k}": v
                                    for k, v in metrics.items()
                                }
                            )
                        wandb.log(metrics, step=metrics["update_steps_0"])
                jax.debug.callback(callback, metrics, original_rng)
            # update rtpt
            jax.debug.callback(rtpt_callback)

            runner_state = (train_states, meta_train_state, tuple(expl_state), test_metrics, test_metrics_modif, rng)

            return runner_state, metrics
        
        def get_test_metrics(train_states, meta_train_state, modif, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):# -> tuple[tuple[Any, Any, Any, Any], tuple[Any, Any, Array, Any]]:
                # this uses the meta-policy to step the environment
                env_state, last_obs, prev_rewards, rng= carry
                rng, rng_a, rng_s = jax.random.split(rng, 3) 
                # 1. get actions of all networks
                # 2. then select the correct ones according to the meta_policy

                def compute_actions(train_state):
                    q_vals = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        last_obs,
                        train=False,
                    )
                    # different eps for each env
                    _rngs = jax.random.split(rng_a, config["TEST_NUM_ENVS"])
                    eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"]) 
                    new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
                    return new_action, q_vals

                all_actions, all_q_vals = jax.vmap(compute_actions)(train_states)

                meta_policy_mode = config.get("META_POLICY", "llm")
                llm_pretrain = config.get("LLM_PRETRAIN", False)
                random_pretrain = config.get("RANDOM_PRETRAIN", False)

                # Combine Q-values depending on mode
                def handle_conditional():
                    active_agent_q_vals = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                    agent_probs = jax.nn.softmax(active_agent_q_vals, axis=-1)  # (128, 3)
                    max_q_vals = jnp.transpose(jnp.max(all_q_vals, axis=-1))    # (128, 3)
                    return agent_probs * max_q_vals                              # (128, 3)

                def handle_llm_pretrain():
                    # llm pretrainining requires access to both, llm meta-policy and combined
                    llm_q= meta_policy_llm(meta_network, meta_train_state, last_obs, env_state)
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

                # Decide which Q-combination strategy to apply
                combined_q = jax.lax.switch(
                    jnp.array([
                        llm_pretrain,
                        random_pretrain,
                        meta_policy_mode == "conditional",
                        True # default if nothing else is true
                    ], dtype=jnp.bool_).argmax(),  # priority order
                    [handle_llm_pretrain, handle_random_pretrain, handle_conditional, handle_default],
                )

                # Now select active agent — greedy vs exploratory
                use_greedy = config.get("META_GREEDY", True)

                def select_greedy(_):
                    return jnp.argmax(combined_q, axis=-1)  # (128,)

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

                # active_agent shape: (num_envs)
                combined_q_vid = combined_q[0]
                active_agent_vid = active_agent[0]
                # select the actions of the active agent 
                new_action = all_actions[active_agent, jnp.arange(config["TEST_NUM_ENVS"])]
                # use the selected actions to step the environment

                new_obs, new_env_state, reward, new_done, info = jax.lax.cond(
                    modif,
                    lambda _: test_env_modif.step(rng_s, env_state, new_action, env_params),
                    lambda _: test_env.step(rng_s, env_state, new_action, env_params),
                    operand=None
                )
                # only select the first value of all arrays of env_state for video generation
                # (env==0)
                # jax.debug.print("cows: {}", new_env_state.env_state.cows.mask.sum())
                env_state_vid = jax.tree.map(lambda x: x[0], new_env_state)
                # jax.debug.print("cows vid: {}", env_state_vid.env_state.cows.mask.sum())
                # remove all_rewards from info (cannot be logged)
                rewards = info.pop("all_rewards")[:, :num_agents] #removes shaped meta-reward

                # select the rewards of the active agent
                mask = jnp.arange(num_agents) == active_agent[:, None]
                active_rewards = jnp.where(mask, rewards, 0)

                # add all_rewards to prev_all_rewards, copy to all_returns
                new_prev_rewards = prev_rewards + active_rewards
                all_returns = new_prev_rewards
                # extend new_done (128,) to match shape of all_returns (128,3)
                # by copying the value n_agents times
                all_done= jnp.repeat(new_done[:, None], num_agents, axis=1)
                # set all_returns to nan where new_done is not True
                all_returns = jnp.where(
                    all_done, all_returns, jnp.nan * jnp.ones_like(all_returns)
                )
                info["active_returns"] = all_returns[jnp.arange(config["TEST_NUM_ENVS"]), active_agent] # (128,)
                for agent_idx in range(num_agents):
                    info[f"active_returns_{agent_idx}"] = all_returns[:, agent_idx]

                # reset prev_all_rewards to current reward where new_done is True
                new_prev_rewards = jnp.where(
                    all_done, active_rewards, new_prev_rewards
                )

                return (new_env_state, new_obs, new_prev_rewards, rng), (info, env_state_vid, active_agent_vid, combined_q_vid, new_done[0])

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

            # return mean of done infos
            # done_infos = jax.tree.map(
            #     lambda x: jnp.nanmean(
            #         jnp.where(
            #             infos["returned_episode"],
            #             x,
            #             jnp.nan,
            #         )
            #     ),
            #     infos,
            # )
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                / infos["returned_episode"].sum(),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_states, meta_train_state, False, _rng)

        rng, _rng = jax.random.split(rng)
        test_metrics_modif = get_test_metrics(train_states, meta_train_state, True, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = env.reset(_rng, env_params)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, meta_train_state, expl_state, test_metrics, test_metrics_modif, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train