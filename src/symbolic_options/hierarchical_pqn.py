import os
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from typing import Any
from rtpt import RTPT

from flax.linen.initializers import constant, orthogonal
import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import wandb
import threading


from jaxtari.wrappers import MultiRewardLogEnvState

class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    num_layers: int = 3
    norm_type: str = "layer_norm"
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
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = normalize(x)
            x = nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)

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


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


video_thread = None
curr_renderer = None

def video_callback(states, dones, step):
    global video_thread
    if video_thread is not None and video_thread.is_alive():
        print("Thread is still running, skipping video generation")
        return
    
    video_thread = threading.Thread(target=collect_video, args=(states, dones, step))
    video_thread.start()

def collect_video(states, dones, step):

    video_folder = f"{wandb.run.dir}/media/videos/"
    os.makedirs(video_folder, exist_ok=True)
    if isinstance(states, MultiRewardLogEnvState):
        states = states.env_state

    # num_states is where the first done is True
    num_states = jnp.argmax(dones)
    # or len of states.obs if no done is True
    if num_states == 0:
        num_states = len(jax.tree_util.tree_leaves(states.obs)[0])

    rasters = jax.vmap(curr_renderer.render)(states)
    # select every 4th frame (and only the first num_states)
    frames = np.array(rasters[:num_states][::4],dtype=np.uint8)
    # shape currently is (N, H, W, 3)
    # but should be (N, 3, H, W)
    frames = np.transpose(frames, (0, 3, 2, 1))
    video = wandb.Video(frames, fps=64, format="mp4")
    wandb.log({f"video_{step}": video}, step=wandb.run.step)

rtpt = None
def rtpt_callback():
    global rtpt
    rtpt.step()

def make_train(config, env, meta_policy, renderer):
    global curr_renderer
    global rtpt
    curr_renderer = renderer
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


    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset)(
        jax.random.split(rng, n_envs)#, env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step#, in_axes=(0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action)#, env_params)

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

    def train(rng):

        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=env.action_space().n,
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
        )

        def create_agent(rng, network):
            obs_len = np.prod(env.observation_space().shape)
            init_x = jnp.zeros(obs_len)
            network_variables = network.init(rng, init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        # TODO: use config value to determine num_agents (shaped / env reward) 
        num_agents = len(env.reward_funcs)
        if config.get("META_SHAPED_REWARD", False):
            num_agents -= 1 # remove one if shaped reward is given

        # create multiple agents
        # networks.append(meta_network)
        rng_keys = jax.random.split(rng, num_agents)
        train_states = jax.vmap(create_agent, in_axes=(0, None))(rng_keys, network)

        meta_policy_string = config.get("META_POLICY", "llm")
        if meta_policy_string == "learned" or meta_policy_string == "combined": 
            meta_network = QNetwork(
                action_dim=num_agents,
                norm_type=config["NORM_TYPE"],
                norm_input=config.get("NORM_INPUT", False),
            )
            meta_train_state = create_agent(rng, meta_network)
        else:
            meta_network = None
            meta_train_state = None

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            
            train_states, meta_train_state, expl_state, test_metrics, rng = runner_state

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

                # active_agent_q_vals = meta_policy(env_state) # (128,3)
                active_agent_q_vals = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                # either select directly or combine with action_q_vals 
                if config.get("META_POLICY", "llm") == "conditional":
                    agent_probs = jax.nn.softmax(active_agent_q_vals, axis=-1) # (128,3)
                    max_q_vals = jnp.max(all_q_vals, axis=-1) # (3, 128)
                    max_q_vals = jnp.transpose(max_q_vals) # (128,3)
                    # combine agent_probs with max q_vals of all agents
                    combined_q = agent_probs * max_q_vals # (128,3)
                else:
                    combined_q = active_agent_q_vals 
                # select active agent with highest q_val * valuation
                # NOTE: instead of greedy, we could also sample from the distribution
                if config.get("META_GREEDY", True): 
                    active_agent = jnp.argmax(combined_q, axis=-1) # (128,)
                else:
                    rng_s, _rng_s = jax.random.split(rng_s)
                    #NOTE: not sure if this is the best way to create the log-distribution 
                    combined_probs = jax.nn.softmax(combined_q, axis=-1) # (128,3)
                    combined_log_probs = jnp.log(combined_probs)
                    active_agent = jax.random.categorical(_rng_s, combined_log_probs, -1)

                # select the q_vals and action of the active agent
                q_vals = all_q_vals[active_agent, jnp.arange(config["NUM_ENVS"]), :]
                new_action = all_actions[active_agent, jnp.arange(config["NUM_ENVS"])] # (128,)

                new_obs, new_env_state, reward, new_done, info = vmap_step(
                    config["NUM_ENVS"]
                )(rng_s, env_state, new_action)

                rewards = info.pop("all_rewards") #(128,3)
                # add reward to end -> (128,4)
                rewards = jnp.concatenate((rewards, reward[:, None]), axis=1)


                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    agent=active_agent,
                    rewards=config.get("REW_SCALE", 1)*rewards,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
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
                last_q = jnp.max(last_q, axis=-1)

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
                    next_q = jnp.max(transition.q_val, axis=-1)
                    return (lambda_returns, next_q), lambda_returns

                last_q = last_q * (1 - transitions.done[-1])
                # lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
                lambda_returns = transitions.rewards[-1, :, state_idx] + config["GAMMA"] * last_q
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree_map(lambda x: x[:-1], transitions),
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
                            q_vals, updates = network.apply(
                                {"params": params, "batch_stats": train_state.batch_stats},
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                            )  # (batch_size*2, num_actions)

                            chosen_action_qvals = jax.lax.cond(
                                state_idx == -1,
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
                    minibatches = jax.tree_util.tree_map(
                        lambda x: preprocess_transition(x, _rng), transitions
                    )  # num_actors*num_envs (batch_size), ...
                    targets = jax.tree_map(
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
                metrics = {
                    "env_step": train_state.timesteps,
                    "update_steps": train_state.n_updates,
                    "env_frame": train_state.timesteps * env.observation_space().shape[-1],
                    "grad_steps": train_state.grad_steps,
                    "td_loss": loss.mean(),
                    "qvals": qvals.mean(),
                }
                return metrics, train_state

            # end of _update_agent
            rngs = jax.random.split(rng, num_agents)
            metrics, train_states = jax.vmap(_update_agent, in_axes=(0, 0, 0, None))(train_states, jnp.arange(num_agents), rngs, network)
            # currently each key has a list of three values, make it s.t. we have key_0, key_1, key_2
            metrics = {f"{k}_{i}": v[i] for k, v in metrics.items() for i in range(v.shape[0])}
            
            meta_policy_string = config.get("META_POLICY", "llm")
            if meta_policy_string == "learned" or meta_policy_string == "combined":
                meta_reward_idx = num_agents # num_agents reward is shaped or env reward
                # note that the reward_idx works, because we add the env_reward to the end of all_rewards
                # so we either select the shaped reward or the env reward
                metrics_meta, meta_train_state = _update_agent(meta_train_state, meta_reward_idx, rng, meta_network)
                metrics.update({f"meta_{k}": v for k, v in metrics_meta.items()})

            metrics.update({k: jnp.nanmean(v) for k, v in infos.items()}),

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_states.n_updates[0] % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"]) == 0,
                    lambda _: get_test_metrics(train_states, meta_train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test_{k}": v for k, v in test_metrics.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_rng)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    # wandb.log(metrics, step=metrics["update_steps"][0])
                    wandb.log(metrics, step=metrics["update_steps_0"])
                jax.debug.callback(callback, metrics, original_rng)
            # update rtpt
            jax.debug.callback(rtpt_callback)

            runner_state = (train_states, meta_train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(train_states, meta_train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                # this uses the meta-policy to step the environment
                env_state, last_obs, prev_rewards, rng= carry
                # 1. get actions of all networks
                # 2. then select the correct ones according to the meta_policy
                
                rng, *rngs = jax.random.split(rng, num_agents+1)
                rngs = jnp.stack(rngs)  # Stack to use in vmap

                # Apply the network to all train_states in parallel
                q_vals = jax.vmap(lambda ts, r: network.apply(
                    {"params": ts.params, "batch_stats": ts.batch_stats}, 
                    last_obs, 
                    train=False
                ))(train_states, rngs)

                # Create an epsilon array
                eps = jnp.full((num_agents, config["TEST_NUM_ENVS"]), config["EPS_TEST"])

                rng_keys = jax.random.split(rng, num_agents * config["TEST_NUM_ENVS"])
                rng_keys = jnp.reshape(rng_keys, (num_agents, config["TEST_NUM_ENVS"], -1))
                # Compute actions in parallel using vmap
                actions = jax.vmap(jax.vmap(eps_greedy_exploration))(
                    rng_keys,
                    q_vals,
                    eps
                )

                # actions shape: (num_agents, num_envs)
                active_agent_q_vals = meta_policy(meta_network, meta_train_state, last_obs, env_state)
                # either select directly or combine with action_q_vals 
                if config.get("META_POLICY", "llm") == "conditional":
                    agent_probs = jax.nn.softmax(active_agent_q_vals, axis=-1) # (128,3)
                    max_q_vals = jnp.max(q_vals, axis=-1) # (3, 128)
                    max_q_vals = jnp.transpose(max_q_vals) # (128,3)
                    # combine agent_probs with max q_vals of all agents
                    combined_q = agent_probs * max_q_vals # (128,3)
                else:
                    combined_q = active_agent_q_vals 

                # select active agent with highest q_val * valuation
                # NOTE: instead of greedy, we could also sample from the distribution
                if config.get("META_GREEDY", True): 
                    active_agent = jnp.argmax(combined_q, axis=-1) # (128,)
                else:
                    rng, _rng_s = jax.random.split(rng)
                    #NOTE: not sure if this is the best way to create the log-distribution 
                    combined_probs = jax.nn.softmax(combined_q, axis=-1) # (128,3)
                    combined_log_probs = jnp.log(combined_probs)
                    active_agent = jax.random.categorical(_rng_s, combined_log_probs, -1)

                # active_agent shape: (num_envs)
                # select the actions of the active agent 
                action = actions[active_agent, jnp.arange(config["TEST_NUM_ENVS"])]

                # use the selected actions to step the environment
                new_obs, new_env_state, reward, done, info = vmap_step(
                    config["TEST_NUM_ENVS"]
                )(_rng, env_state, action)
                # only select the first value of all arrays of env_state for video generation
                # (env==0)
                env_state_vid = jax.tree_map(lambda x: x[0], new_env_state)
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
                all_done= jnp.repeat(done[:, None], num_agents, axis=1)
                # set all_returns to nan where new_done is not True
                all_returns = jnp.where(
                    all_done, all_returns, jnp.nan * jnp.ones_like(all_returns)
                )
                info["active_returns"] = all_returns[jnp.arange(config["NUM_ENVS"]), active_agent] # (128,)
                for agent_idx in range(num_agents):
                    info[f"active_returns_{agent_idx}"] = all_returns[:, agent_idx]

                # reset prev_all_rewards to current reward where new_done is True
                new_prev_rewards = jnp.where(
                    all_done, active_rewards, new_prev_rewards
                )

                return (new_env_state, new_obs, new_prev_rewards, rng), (info, env_state_vid, done[0])

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = vmap_reset(config["TEST_NUM_ENVS"])(_rng)

            init_rewards = jnp.zeros((config["NUM_ENVS"], num_agents))
            _, output = jax.lax.scan(
                _env_step, (env_state, init_obs, init_rewards, _rng), None, config["TEST_NUM_STEPS"]
            )
            infos, states, dones = output

            if config.get("RECORD_VIDEO", False):
                jax.lax.cond(
                    train_states.n_updates[0] > 0,
                    lambda _: jax.debug.callback(video_callback, states, dones, train_states.n_updates[0]),
                    lambda _: None,
                    operand=None,
                )

            # return mean of done infos
            done_infos = jax.tree_map(
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
        test_metrics = get_test_metrics(train_states, meta_train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = vmap_reset(config["NUM_ENVS"])(_rng)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, meta_train_state, expl_state, test_metrics, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train