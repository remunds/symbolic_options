import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95" # limit GPU memory usage to 95%
import time
import copy
import jax
import jax.numpy as jnp
import wandb
import hydra
from omegaconf import OmegaConf
from jaxatari.games.jax_seaquest import JaxSeaquest, SeaquestRenderer 
from jaxatari.games.jax_kangaroo import JaxKangaroo, KangarooRenderer
from symbolic_options.hierarchical_pqn_jaxtari import make_train as make_train_hier_jaxatari
# from symbolic_options.hierarchical_pqn_jaxtari_few_shot import make_train as make_train_hier_jaxatari
from symbolic_options.pqn_jaxtari import make_train as make_train_pqn_jaxatari
from symbolic_options.pqn_jaxtari_img import make_train as make_train_pqn_jaxatari_img


from jaxatari.wrappers import ObjectCentricWrapper, FlattenObservationWrapper, AtariWrapper, PixelObsWrapper



def _build_eval_envs(create_env_fn, config):
    """Build a tuple of (env_obj, label) pairs from the eval_envs config list."""
    eval_envs_list = []
    for entry in config.get("eval_envs", []):
        label = entry.get("label", "default")
        kwargs = {k: v for k, v in entry.items() if k != "label"}
        env_obj = create_env_fn(train=False, **kwargs)
        eval_envs_list.append((env_obj, label))
    return tuple(eval_envs_list)


def outer_make_train(config):
    noisy_training = config.get("NOISY_TRAINING", False)
    detection_probability = config.get("DETECT_PROB", 1.0)
    noise_std_dev = config.get("NOISE_STD_DEV", 0.0)

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
        def create_env(train: bool = False, randomized_enemy: bool = False, lazy_enemy: bool = False, noisy_eval: bool = False):
            env = JaxPong(reward_funcs=reward_funcs)
            if randomized_enemy:
                env = RandomizedEnemyWrapper(env)
            if lazy_enemy:
                env = LazyEnemyWrapper(env)
            if train:
                env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=detection_probability if noisy_eval else 1.0, std_dev=noise_std_dev if noisy_eval else 0.0)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False, False) 
        eval_envs = _build_eval_envs(create_env, config) 
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
        def create_env(train: bool = False, left_drift: bool = False, small_paddle: bool = False, noisy_eval: bool = False):
            env = JaxBreakout(reward_funcs=reward_funcs)
            if left_drift:
                env = SpeedMode(env)
            if small_paddle:
                env = BigPaddle(env)
            if train:
                env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life)
            else:
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=detection_probability if noisy_eval else 1.0, std_dev=noise_std_dev if noisy_eval else 0.0)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False, False)
        eval_envs = _build_eval_envs(create_env, config) 
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
        def create_env(train: bool = False, stop_all_cars: bool = False, stop_random_cars: bool = False, speed_mode: bool = False, noisy_eval: bool = False): 
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
                env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=detection_probability if noisy_eval else 1.0, std_dev=noise_std_dev if noisy_eval else 0.0)
            env = ObjectCentricWrapper(env)
            env = FlattenObservationWrapper(env)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False, False, False) # train on default
        eval_envs = _build_eval_envs(create_env, config) 
        renderer = FreewayRenderer()


    elif config.get("ENV_NAME", None) == "Seaquest":
        from symbolic_options.reward_functions.seaquest import collect_divers_reward, fight_enemies_reward, upward_reward, shaped_reward, env_reward, total_rescued, total_collected, total_shot, total_surface_without_dying 
        from symbolic_options.reward_functions.seaquest import learned_meta_policy, llm_meta_policy, combined_meta_policy 
        from symbolic_options.reward_functions.seaquest import shoot_default_policy as conditional_meta_policy #conditional_meta_policy
        from jaxatari.wrappers import MultiRewardLogWrapper
        from jaxatari.games.mods.seaquest_mods import DisableEnemiesWrapper
        # NOTE: the order of the rewards needs to align with the LLM-based meta-policy
        # NOTE: if conditional or combined provide idle_reward (not necessary for llm and learned)
        # this makes sure that there is always a fallback if no rule evaluates to true
        reward_funcs = [fight_enemies_reward, collect_divers_reward, upward_reward, total_rescued, total_collected, total_shot, total_surface_without_dying]

        # This tests generated rewards
        if config.get("MODEL", None) == "gemini_pro":
            print("Using Gemini Pro reward functions")
            from symbolic_options.reward_functions.seaquest_gemini_pro import reward_rescue_state, reward_combat_survival_state, reward_surface_state
            #TODO:
            from symbolic_options.reward_functions.seaquest_gemini_pro import llm_meta_policy 
            from jaxatari.wrappers import MultiRewardLogWrapper
            from jaxatari.games.mods.seaquest_mods import DisableEnemiesWrapper
            reward_funcs = [reward_combat_survival_state, reward_rescue_state, reward_surface_state, total_rescued, total_collected, total_shot, total_surface_without_dying]

        if config.get("MODEL", None) == "codex":
            print("Using codex reward functions")
            #TODO:
            from symbolic_options.reward_functions.seaquest_codex import reward_safe_navigation_state, reward_collect_divers_state, reward_surface_state, reward_oxygen_state, reward_opportunistic_combat_state 
            from symbolic_options.reward_functions.seaquest_codex import llm_meta_policy 
            from jaxatari.wrappers import MultiRewardLogWrapper
            from jaxatari.games.mods.seaquest_mods import DisableEnemiesWrapper
            reward_funcs = [reward_opportunistic_combat_state, reward_collect_divers_state, reward_safe_navigation_state, reward_surface_state, reward_oxygen_state, total_rescued, total_collected, total_shot, total_surface_without_dying]

        if config.get("NO_REWARDS", False):
            # reward_funcs = [env_reward, env_reward, env_reward] #use env_reward for all options
            reward_funcs = [env_reward, env_reward, env_reward, total_rescued, total_collected, total_shot, total_surface_without_dying]

        if config.get("SHAPED_REWARD", False):
            from symbolic_options.reward_functions.seaquest import shaped_reward
            if config.get("SHAPED_REWARD_SIMPLE", False):
                from symbolic_options.reward_functions.seaquest import shaped_reward_simple
                shaped_reward = shaped_reward_simple
            if config.get("SHAPED_REWARD_LLM", False):
                from symbolic_options.reward_functions.seaquest import shaped_reward_llm
                shaped_reward = shaped_reward_llm
            reward_funcs = [shaped_reward, env_reward, env_reward, total_rescued, total_collected, total_shot, total_surface_without_dying]

        # Shaped reward is reward function for meta-policy (not necessary, if meta-policy does not learn) 
        if config.get("META_SHAPED_REWARD", False):
            reward_funcs.append(shaped_reward)


        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        def create_env(train: bool = False, no_enemies: bool = False, noisy_eval: bool = False):
            env = JaxSeaquest(reward_funcs=reward_funcs)
            if no_enemies:
                env = DisableEnemiesWrapper(env)
            if train:
                if noisy_training:
                    env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life, detect_prob=detection_probability, std_dev=noise_std_dev)
                else:
                    env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life, detect_prob=1.0, std_dev=0.0)
            else:
                if noisy_eval:
                    env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=detection_probability, std_dev=noise_std_dev)
                else:
                    env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=1.0, std_dev=0.0)
                # env = AtariWrapper(env, sticky_actions=False, episodic_life=False)
            if config.get("OBJECT_CENTRIC", True):
                env = ObjectCentricWrapper(env)
                env = FlattenObservationWrapper(env)
            else:
                env = PixelObsWrapper(env, do_pixel_resize=True, pixel_resize_shape=(84, 84), grayscale=True)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False)
        eval_envs = _build_eval_envs(create_env, config) 
        renderer = SeaquestRenderer()
    elif config.get("ENV_NAME", None) == "Kangaroo":
        from symbolic_options.reward_functions.kangaroo import navigate_reward, handle_enemies_reward, collect_fruits_reward, env_reward, reached_platform_level, enemies_killed, fruits_collected 
        # from symbolic_options.reward_functions.kangaroo import obstacle_avoidance_reward, vertical_navigation_reward, fruit_collection_reward
        from symbolic_options.reward_functions.kangaroo import llm_meta_policy, learned_meta_policy, combined_meta_policy, conditional_meta_policy
        from jaxatari.wrappers import MultiRewardLogWrapper#, MultiRewardWrapper
        from jaxatari.games.mods.kangaroo_mods import DisableThreadsWrapper 

        reward_funcs = [navigate_reward, handle_enemies_reward, collect_fruits_reward, reached_platform_level, enemies_killed, fruits_collected] 
        # reward_funcs = [vertical_navigation_reward, obstacle_avoidance_reward, fruit_collection_reward, reached_platform_level, enemies_killed, fruits_collected]
        if config.get("NO_REWARDS", False):
            # reward_funcs = [env_reward, env_reward, env_reward] #use env_reward for all options
            reward_funcs = [env_reward, env_reward, env_reward, reached_platform_level, enemies_killed, fruits_collected] 

        if config.get("SHAPED_REWARD", False):
            from symbolic_options.reward_functions.kangaroo import shaped_reward
            if config.get("SHAPED_REWARD_SIMPLE", False):
                from symbolic_options.reward_functions.kangaroo import shaped_reward_simple
                shaped_reward = shaped_reward_simple
            if config.get("SHAPED_REWARD_LLM", False):
                from symbolic_options.reward_functions.kangaroo import shaped_reward_llm
                shaped_reward = shaped_reward_llm
            reward_funcs = [shaped_reward, env_reward, env_reward, reached_platform_level, enemies_killed, fruits_collected]

        sticky_actions = config.get("STICKY_ACTIONS", False)
        episodic_life = config.get("EPISODIC_LIFE", False)
        def create_env(train=False, no_enemies: bool = False, noisy_eval: bool = False):
            env = JaxKangaroo(reward_funcs=reward_funcs)
            if no_enemies:
                env = DisableThreadsWrapper(env)
            if train:
                if noisy_training:
                    env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life, detect_prob=detection_probability, std_dev=noise_std_dev, first_fire=False)
                else:
                    env = AtariWrapper(env, sticky_actions=sticky_actions, episodic_life=episodic_life, detect_prob=1.0, std_dev=0.0, first_fire=False)
            else:
                if noisy_eval:
                    env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=detection_probability, std_dev=noise_std_dev, first_fire=False)
                else:
                    env = AtariWrapper(env, sticky_actions=False, episodic_life=False, detect_prob=1.0, std_dev=0.0, first_fire=False)
            if config.get("OBJECT_CENTRIC", True):
                env = ObjectCentricWrapper(env)
                env = FlattenObservationWrapper(env)
            else:
                env = PixelObsWrapper(env, do_pixel_resize=True, pixel_resize_shape=(84, 84), grayscale=True)
            env = MultiRewardLogWrapper(env)
            return env
        env = create_env(True, False)
        eval_envs = _build_eval_envs(create_env, config) 
        renderer = KangarooRenderer()
    # only quick PQN test:
    elif "Craftax" in config.get("ENV_NAME", None):
        from craftax.craftax_env import make_craftax_env_from_name
        from symbolic_options.purejaxql.craftax_wrappers import (
            LogWrapper,
            OptimisticResetVecEnvWrapper,
            BatchEnvWrapper,
            MultiRewardWrapper
        )
        # from symbolic_options.reward_functions.craftax import llm_meta_policy, learned_meta_policy, combined_meta_policy, conditional_meta_policy
        from symbolic_options.reward_functions.craftax_classic import llm_meta_policy, learned_meta_policy, combined_meta_policy, conditional_meta_policy, llm_meta_policy_rnn, conditional_meta_policy_rnn, learned_meta_policy_rnn, combined_meta_policy_rnn
        from symbolic_options.purejaxql.craftax_wrappers import MultiRewardLogWrapper, LogWrapper, NoNecessitiesWrapper, ExplorationMapWrapper
        from symbolic_options.reward_functions.craftax_classic import survival_reward, combat_reward, resource_collection_reward, crafting_reward, explore, env_reward
        from symbolic_options.utils.video_recorder import CraftaxClassicRenderer

        # reward_funcs_craftax = [survival_reward, combat_reward, resource_collection_reward, crafting_reward, level_progression_reward, explore]
        reward_funcs = [survival_reward, combat_reward, crafting_reward, resource_collection_reward, explore]
        if config.get("NO_REWARDS", False):
            reward_funcs = [env_reward, env_reward, env_reward, env_reward, env_reward] #use env_reward for all options
        # reward_funcs = []
        # renderer = CraftaxRenderer()
        renderer = CraftaxClassicRenderer()
        basic_env = make_craftax_env_from_name(
            config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
        )
        env_params = basic_env.default_params

        def create_env(env, train: bool = False, modification: bool = False):
            if modification:
                env = NoNecessitiesWrapper(env)
            env = ExplorationMapWrapper(env) 
            if len(reward_funcs) > 0:
                env = MultiRewardWrapper(env, reward_funcs)
                env = MultiRewardLogWrapper(env)
            else:
                env = LogWrapper(env)

            if train:
                if config["USE_OPTIMISTIC_RESETS"]:
                    env = OptimisticResetVecEnvWrapper(
                        env,
                        num_envs=config["NUM_ENVS"],
                        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
                    )
                else:
                    env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])
            else:
                if config["USE_OPTIMISTIC_RESETS"]:
                    env = OptimisticResetVecEnvWrapper(
                        env,
                        num_envs=config["TEST_NUM_ENVS"],
                        reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["TEST_NUM_ENVS"]),
                    )
                else:
                    env = BatchEnvWrapper(env, num_envs=config["TEST_NUM_ENVS"])
            return env

        env = create_env(basic_env, True, False)
        test_env = create_env(basic_env, False, False)
        test_env_modif = create_env(basic_env, False, True)

    else:
        raise NotImplementedError(f"Env {config['ENV_NAME']} not implemented.")

    if config.get("HIERARCHICAL", False):
        meta_policy_string = config.get("META_POLICY", "llm")
        if meta_policy_string == "llm":
            if config.get("USE_RNN", False):
                meta_policy = llm_meta_policy_rnn
            else:
                meta_policy = llm_meta_policy
        elif meta_policy_string == "learned":
            if config.get("USE_RNN", False):
                meta_policy = learned_meta_policy_rnn
            else:
                meta_policy = learned_meta_policy
        elif meta_policy_string == "conditional":
            if config.get("USE_RNN", False):
                meta_policy = conditional_meta_policy_rnn
            else:
                meta_policy = conditional_meta_policy
        elif meta_policy_string == "combined":
            if config.get("USE_RNN", False):
                meta_policy = combined_meta_policy_rnn
            else:
                meta_policy = combined_meta_policy
        else:
            raise ValueError("Invalid meta policy")
    else:
        meta_policy = None

    if "Craftax" in config.get("ENV_NAME", None):
        from symbolic_options.hierarchical_pqn_craftax import make_train as make_train_hier_craftax
        from symbolic_options.pqn_craftax import make_train as make_train_pqn_craftax
        from symbolic_options.pqn_rnn_craftax import make_train as make_train_pqn_rnn_craftax
        from symbolic_options.hierarchical_pqn_rnn_craftax import make_train as make_train_hier_rnn_craftax
        if config.get("HIERARCHICAL", False):
            if config.get("USE_RNN", False):
                return make_train_hier_rnn_craftax(config, env, test_env, test_env_modif, env_params, meta_policy, llm_meta_policy, renderer)
            else:
                return make_train_hier_craftax(config, env, test_env, test_env_modif, env_params, meta_policy, llm_meta_policy, renderer)
        else:
            if config.get("USE_RNN", False):
                return make_train_pqn_rnn_craftax(config, env, test_env, test_env_modif, env_params, meta_policy, renderer)
            else:
                return make_train_pqn_craftax(config, env, test_env, test_env_modif, env_params, meta_policy, renderer)
    else:
        if config.get("HIERARCHICAL", False):
            return make_train_hier_jaxatari(config, env, eval_envs, meta_policy, llm_meta_policy, renderer)
        else:
            if not config.get("OBJECT_CENTRIC", True):
                print("Using pixel-based observations and convolutional networks.")
                return make_train_pqn_jaxatari_img(config, env, eval_envs, meta_policy, renderer)
            return make_train_pqn_jaxatari(config, env, eval_envs, meta_policy, renderer)
        
def load_network_params(config):
    if config.get("LOAD_PARAMS", False) and config.get("LOAD_PATH", None) is not None:
        from symbolic_options.purejaxql.save_load import load_params
        import os
        alg_name = config["ALG_NAME"]
        env_name = config["ENV_NAME"]
        load_dir = os.path.join(config["LOAD_PATH"], env_name)
        params = []
        batch_stats = []
        for i in range(config["NUM_SEEDS"]):
            load_path = os.path.join(
                load_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            inner_params = None
            inner_bs = None
            # check if file exists
            if os.path.exists(load_path):
                print(f"Loading params from {load_path}")
                inner_params = load_params(load_path)
                inner_bs = load_params(load_path.replace(".safetensors", "_bs.safetensors"))
            else:
                print(f"File {load_path} does not exist, using random params")
            params.append(inner_params)
            batch_stats.append(inner_bs)

        # transform to pytree
        params = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *params
        )
        batch_stats = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *batch_stats
        )

        return params, batch_stats

    return None, None

def single_run(config):#
    config = {**config, **config["alg"]}

    alg_name = config.get("ALG_NAME", "pqn")
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
    params, batch_stats = load_network_params(config)
    train_vjit = jax.jit(jax.vmap(outer_make_train(config)))
    # time compilation
    print("Compiling...")
    start = time.time()
    # with jax.profiler.trace("outputs/jax-trace", create_perfetto_link=True):
    train_vjit.lower(rngs, params, batch_stats).compile()
    print(f"Compilation took {time.time()-start} seconds.")
    outs = jax.block_until_ready(train_vjit(rngs, params, batch_stats))
    print(f"Took {time.time()-t0} seconds to complete.")

    if config.get("SAVE_PATH", None) is not None:
        from symbolic_options.purejaxql.save_load import save_params
        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_util.tree_map(lambda x: x[i], model_state.params)
            batch_stats = jax.tree_util.tree_map(lambda x: x[i], model_state.batch_stats)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_path_bs = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}_bs.safetensors',
            )
            save_params(params, save_path)
            save_params(batch_stats, save_path_bs)
            print(f"Saved params to {save_path}")

    wandb.finish()


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    print("Running hyperparameter tuning with wandb...")
    default_config = {**default_config, **default_config["alg"]}
    print(default_config)
    alg_name = default_config.get("ALG_NAME", "pqn")
    env_name = default_config["ENV_NAME"]

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        params, batch_stats = load_network_params(config)
        train_vjit = jax.jit(jax.vmap(outer_make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs, params, batch_stats))
        wandb.finish()


    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test/returned_episode_env_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "min": 0.00001,
                "max": 0.001,
            },
            "EPS_FINISH": {
                "min": 0.001,
                "max": 0.1,
            },
            "EPS_DECAY": {
                "min": 0.1,
                "max": 0.7,
            },
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=100)


@hydra.main(version_base=None, config_path="./src/symbolic_options/config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main(None)
