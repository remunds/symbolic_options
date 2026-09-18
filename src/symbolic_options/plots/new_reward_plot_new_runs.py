"""Skill-reward plots for JAXAtari Seaquest and Kangaroo using the newest runs.

Same ``multi_reward_plot`` setup as ``plots.ipynb`` (cell 12) / ``new_reward_plot.py``.
The grid is 2x4: three skill-return panels per game plus a fourth, rightmost
column showing the environment (episode) return curve.

Data sources:
  * NEXUS / baseline methods -- ``raban-emunds-tu-darmstadt/NEXUS_noisy_fix_scaled``,
    selected live from wandb: all runs with ``NOISY_TRAINING=False`` for the five
    methods ``pqn``, ``pqn_hier_baseline``, ``pqn_hier``, ``pqn_hier_llm`` and
    ``pqn_hier_comb`` (2 runs each, merged across their 6 seeds into one curve).
    These runs log eval under ``test_clean`` / ``test_mod`` (not ``test`` /
    ``test_modif``), which the loader handles via ``test_prefix``/``modif``.
  * PPO baseline -- absent from ``NEXUS_noisy_fix_scaled``, so kept from the old
    ``raban-emunds-tu-darmstadt/LENS`` project (same run ids as ``plots.ipynb``).
  * Shaped-reward ablations -- ``raban-emunds-tu-darmstadt/NEXUS_rebuttal2``
    (unchanged), three flat PQN + shaped-reward runs per game:
      - ``pqn_shapedreward``        : option rewards weighted by LLM Q-values
      - ``pqn_shapedreward_simple`` : unweighted sum of option rewards
      - ``pqn_shapedreward_llm``    : LLM-authored dense shaping

Produces:
    new_reward_plot_new_runs.pdf
    new_reward_plot_new_runs.svg
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv
from tueplots import bundles

plt.rcParams.update({"figure.dpi": 150})

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# get data - download and store locally (plots.ipynb cell 1)
# ---------------------------------------------------------------------------
load_dotenv(OUT_DIR / ".env")
api_key = os.getenv("WANDB_API_KEY")
api = wandb.Api(api_key=api_key)


# ---------------------------------------------------------------------------
# plots.ipynb cell 2
# ---------------------------------------------------------------------------
def get_algorithm_from_name(name):
    if "ppo" in name:
        return "PPO (baseline)"
    elif "baseline" in name:
        return "HPQN (baseline)"
    elif "pre" in name:
        return "Pretrained"
    elif "comb" in name:
        # return "Soft meta-policy"
        # return "NEXUS (neurosym.)"
        return "NEXUS (nesy)"
    elif "llm" in name:
        # return "Fixed meta-policy"
        return "NEXUS (symbolic)"
    elif "hier" in name:
        return "NEXUS (neural)"
    elif "pqn" in name:
        return "PQN (baseline)"
    elif "nudge" in name:
        return "Nudge (baseline)"
    elif "blendrl" in name:
        return "BlendRL (baseline)"
    else:
        return "Unknown"


# Additional NEXUS_rebuttal2 shaped-reward ablations. Their run names all
# contain "pqn" (and "_llm_" matches the "llm" branch above), so they need
# explicit display labels instead of get_algorithm_from_name.
NEW_LABELS = {
    "pqn_shapedreward": "PQN + shaped (LLM-weighted)",
    "pqn_shapedreward_simple": "PQN + shaped (uniform)",
    "pqn_shapedreward_llm": "PQN + shaped (LLM-designed)",
}


def _new_label(algo):
    """Display label for a new shaped-reward run, else None."""
    for key in sorted(NEW_LABELS, key=len, reverse=True):
        if algo == key or algo.startswith(key + "_"):
            return NEW_LABELS[key]
    return None


def algo_label(algo):
    return _new_label(algo) or get_algorithm_from_name(algo)


def is_new_ablation(algo):
    return _new_label(algo) is not None


# ---------------------------------------------------------------------------
# plots.ipynb cell 3
# ---------------------------------------------------------------------------
def get_run_data(runs, craftax=False, test_prefix="test", modif="test_modif"):
    data = []
    for run_id in runs:
        run = api.run(runs[run_id])
        print("quering run: ", run.id)
        env_name = run.name.split("_")[-1]
        alg_name = run.name.split("_")[0]
        run_data = run.history()
        seeds = list(set([key.split("/")[0] for key in run_data.keys() if "rng" in key]))
        if env_name.lower() == "kangaroo" and alg_name.lower() == "ppo":
            # remove the seed with rng1797259609, as it is a weird outlier
            seeds = [seed for seed in seeds if seed != "rng1797259609"]

        key_list = [key for key in run_data.keys() if "test" in key]
        retries = 0
        while len(key_list) == 0:
            if retries > 10:
                print("No test keys found after 10 retries, breaking...")
                break
            print("No test keys found, retrying...")
            run = api.run(runs[run_id])
            run_data = run.history()
            key_list = [key for key in run_data.keys() if "test" in key]
            retries += 1

        env_key = "env_step" if "hier" not in run.name else "env_step_0"
        x_data = run_data[env_key].to_numpy()

        def get_seed_data(seed):
            env_ret = run_data[seed + "/returned_episode_env_returns"].to_numpy()
            test_env_ret = run_data[seed + f"/{test_prefix}/returned_episode_env_returns"].to_numpy()
            modif_test_env_ret = run_data[seed + f"/{modif}/returned_episode_env_returns"].to_numpy()
            if craftax:
                score = run_data[seed + "/score"].to_numpy()
                test_score = run_data[seed + "/test/score"].to_numpy()

            opt_rets = []
            match env_name.lower():
                case "pong":
                    option_strs = ["returned_episode_returns_0", "returned_episode_returns_1", "returned_episode_returns_2"]
                case "breakout":
                    option_strs = ["returned_episode_returns_0", "returned_episode_returns_1", "returned_episode_returns_2"]
                case "freeway":
                    option_strs = ["returned_episode_returns_0", "returned_episode_returns_1"]
                case "kangaroo":
                    option_strs = ["returned_episode_returns_0", "returned_episode_returns_1", "returned_episode_returns_2", "returned_episode_returns_3", "returned_episode_returns_4", "returned_episode_returns_5"]
                case "seaquest":
                    option_strs = ["returned_episode_returns_0", "returned_episode_returns_1", "returned_episode_returns_2", "returned_episode_returns_3", "returned_episode_returns_4", "returned_episode_returns_5", "returned_episode_returns_6"]
                case "craftax":
                    option_strs = ["returned_episode_returns_0", "returned_episode_returns_1", "returned_episode_returns_2"]
            if craftax:
                option_strs = ["returned_episode_returns_0", "returned_episode_returns_1", "returned_episode_returns_2", "returned_episode_returns_3", "returned_episode_returns_4"]
            for opt_str in option_strs:
                ret = run_data[seed + "/" + opt_str].to_numpy()
                test_ret = run_data[seed + f"/{test_prefix}/" + opt_str].to_numpy()
                if not craftax:
                    modif_test_ret = run_data[seed + f"/{modif}/" + opt_str].to_numpy()
                    opt_rets += [ret, test_ret, modif_test_ret]
                else:
                    opt_rets += [ret, test_ret]
            if not craftax:
                return env_ret, test_env_ret, modif_test_env_ret, *opt_rets
            else:
                return env_ret, test_env_ret, modif_test_env_ret, score, test_score, *opt_rets

        all_data = np.array([get_seed_data(seed) for seed in seeds])
        print(all_data.shape)

        if not craftax:
            row = {
                "run_key": run_id,
                "run_id": run.id,
                "run_name": run.name.lower(),
                "seeds": seeds,
                "algorithm": get_algorithm_from_name(run.name),
                "x_data": x_data,
                "env_ret": all_data[:, 0],
                "test_env_ret": all_data[:, 1],
                "modif_test_env_ret": all_data[:, 2],
            }
            for i in range(3, all_data.shape[1]):
                if i % 3 == 0:
                    opt_num = (i // 3) - 1
                    row[f"opt{opt_num}_ret"] = all_data[:, i]
                    row[f"test_opt{opt_num}_ret"] = all_data[:, i + 1]
                    row[f"modif_test_opt{opt_num}_ret"] = all_data[:, i + 2]
        else:
            row = {
                "run_key": run_id,
                "run_id": run.id,
                "run_name": run.name.lower(),
                "seeds": seeds,
                "algorithm": get_algorithm_from_name(run.name),
                "x_data": x_data,
                "env_ret": all_data[:, 0],
                "test_env_ret": all_data[:, 1],
                "modif_test_env_ret": all_data[:, 2],
                "score": all_data[:, 3],
                "test_score": all_data[:, 4],
                "opt0_ret": all_data[:, 5],
                "test_opt0_ret": all_data[:, 6],
                "opt1_ret": all_data[:, 7],
                "test_opt1_ret": all_data[:, 8],
                "opt2_ret": all_data[:, 9],
                "test_opt2_ret": all_data[:, 10],
                "opt3_ret": all_data[:, 11],
                "test_opt3_ret": all_data[:, 12],
                "opt4_ret": all_data[:, 13],
                "test_opt4_ret": all_data[:, 14],
            }

        data.append(row)
    df = pd.DataFrame(data)
    return df


# ---------------------------------------------------------------------------
# Previous JAXAtari runs (plots.ipynb cell 5)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# NEXUS methods: newest runs from NEXUS_noisy_fix_scaled (NOISY_TRAINING=False)
# ---------------------------------------------------------------------------
NEXUS_PROJECT = "raban-emunds-tu-darmstadt/NEXUS_noisy_fix_scaled"
# This project logs eval under test_clean / test_mod (not test / test_modif).
NEXUS_TEST_PREFIX = "test_clean"
NEXUS_MODIF = "test_mod"


def select_clean_runs(project, env_name):
    """All finished, NOISY_TRAINING=False runs of *project* for *env_name*."""
    runs = {}
    for run in api.runs(project):
        alg = run.config.get("alg", {})
        if alg.get("ENV_NAME") != env_name:
            continue
        if alg.get("NOISY_TRAINING", False):
            continue
        if run.state != "finished":
            continue
        runs[run.id] = f"{project}/runs/{run.id}"
    return runs


def aggregate_multi_seeds(rows):
    """Merge several runs of the same method by concatenating their seeds.

    Verbatim from plots.ipynb cell 9.
    """
    run_name = rows.iloc[0]["run_name"]
    rows_dict = rows.to_dict("records")
    assert all(row["run_name"] == run_name for row in rows_dict)
    assert all(np.array_equal(row["x_data"], rows.iloc[0]["x_data"]) for row in rows_dict)

    col_names = rows.columns.difference(["run_id", "run_key", "run_name", "seeds", "algorithm", "x_data"])
    aggregated_row = {
        "run_id": "+".join(row["run_id"] for row in rows_dict),
        "run_name": run_name,
        "seeds": [row["seeds"][0] for row in rows_dict],
        "algorithm": rows.iloc[0]["algorithm"],
        "x_data": rows.iloc[0]["x_data"],
    }
    for col in col_names:
        aggregated_row[col] = np.concatenate([row[col] for row in rows_dict], axis=0)
    return aggregated_row


def aggregate_runs_by_name(df):
    return pd.DataFrame([aggregate_multi_seeds(grp) for _, grp in df.groupby("run_name", sort=False)])


# plots.ipynb cell 7: Seaquest's comb runs were named '*_shootdefault'; strip the
# suffix so they merge into plain 'pqn_hier_comb_seaquest'.
def strip_shootdefault(df):
    df["run_name"] = df["run_name"].str.replace("_shootdefault", "", case=False)
    return df


nexus_seaquest_df = aggregate_runs_by_name(strip_shootdefault(get_run_data(
    select_clean_runs(NEXUS_PROJECT, "Seaquest"),
    test_prefix=NEXUS_TEST_PREFIX, modif=NEXUS_MODIF,
)))
nexus_kangaroo_df = aggregate_runs_by_name(get_run_data(
    select_clean_runs(NEXUS_PROJECT, "Kangaroo"),
    test_prefix=NEXUS_TEST_PREFIX, modif=NEXUS_MODIF,
))

# ---------------------------------------------------------------------------
# Shaped-reward ablations (NEXUS_rebuttal2, flat PQN + shaped reward)
# ---------------------------------------------------------------------------
seaquest_shaped_run_ids = {
    "shaped": "i9qzgdge",
    "shaped_simple": "w9dw1bgk",
    "shaped_llm": "1hkxcsdn",
}
kangaroo_shaped_run_ids = {
    "shaped": "80bgyq4o",
    "shaped_simple": "vxx1sdvy",
    "shaped_llm": "8sgqi5ru",
}
SHAPED_PROJECT = "raban-emunds-tu-darmstadt/NEXUS_rebuttal2"
seaquest_shaped_runs = {
    run_id: f"{SHAPED_PROJECT}/runs/{run_hash}" for run_id, run_hash in seaquest_shaped_run_ids.items()
}
kangaroo_shaped_runs = {
    run_id: f"{SHAPED_PROJECT}/runs/{run_hash}" for run_id, run_hash in kangaroo_shaped_run_ids.items()
}
shaped_seaquest_df = get_run_data(seaquest_shaped_runs)
shaped_kangaroo_df = get_run_data(kangaroo_shaped_runs)

# ---------------------------------------------------------------------------
# PPO baseline: absent from NEXUS_noisy_fix_scaled, so kept from the old LENS
# project (same run ids as plots.ipynb). LENS logs eval under test/test_modif,
# i.e. the loader defaults.
# ---------------------------------------------------------------------------
LENS_PROJECT = "raban-emunds-tu-darmstadt/LENS"
ppo_seaquest_df = get_run_data({"ppo": f"{LENS_PROJECT}/runs/dfc6vzov"})
ppo_kangaroo_df = get_run_data({"ppo": f"{LENS_PROJECT}/runs/356qhfba"})

# Concatenate NEXUS + PPO + shaped rows into the same per-game dataframe.
seaquest_df = pd.concat([ppo_seaquest_df, nexus_seaquest_df, shaped_seaquest_df], ignore_index=True)
kangaroo_df = pd.concat([ppo_kangaroo_df, nexus_kangaroo_df, shaped_kangaroo_df], ignore_index=True)


# ---------------------------------------------------------------------------
# plots.ipynb cell 12
# ---------------------------------------------------------------------------
def single_reward_plot(game_data, algos, reward_row, ax, opt_name):
    # plot the reward (e.g. collect_divers) of all algos
    colors = sns.color_palette("colorblind")
    baselines = ["PPO (baseline)", "PQN (baseline)", "HPQN (baseline)", "Nudge (baseline)", "BlendRL (baseline)"]
    non_baselines = [algo for algo in algos if algo_label(algo) not in baselines]
    colors_baseline = sns.color_palette("gray", n_colors=len(baselines))
    for algo in algos:
        row = game_data[game_data["run_name"] == algo].iloc[0]
        x_data = row["x_data"]
        means = np.nanmean(row[f"{reward_row}_ret"], axis=0)
        stds = np.nanstd(row[f"{reward_row}_ret"], axis=0)
        x_data = x_data[:len(means)]
        alg_label = algo_label(algo)
        if alg_label in baselines:
            color = colors_baseline[baselines.index(alg_label)]
        else:
            color = colors[non_baselines.index(algo)]
        ax.plot(x_data, means, label=alg_label, color=color, zorder=1,
                linestyle="--" if is_new_ablation(algo) else "-")
        ax.fill_between(x_data, means - stds, means + stds, color=color, alpha=0.2, zorder=0)
        ax.set_title(f"{opt_name}")
    ax.set_xlabel("Steps")


def multi_reward_plot(df, opt_names, axes):
    # plot all single_reward_plots for all envs
    game_name = df["run_name"].iloc[0].split("_")[-1]
    opts = ["opt3", "opt4", "opt5"]
    if "sea" in game_name:
        opts = ["opt4", "opt5", "opt6"]
    if "craft" in game_name:
        opts = ["opt0", "opt1", "opt2", "opt3", "opt4"]
    if "freeway" in game_name:
        opts = ["opt0", "opt1"]
    if "breakout" in game_name:
        opts = ["opt0", "opt1", "opt2"]
    if "pong" in game_name:
        opts = ["opt0", "opt1", "opt2"]
    if "craft" in game_name:
        algos = [f"pqn_rnn_{game_name}", f"pqn_rnn_hier_{game_name}", f"pqn_rnn_hier_llm_{game_name}", f"pqn_rnn_hier_comb_{game_name}"]
    elif "freeway" in game_name:
        algos = [f"pqn_{game_name}", f"pqn_hier_baseline_{game_name}", f"pqn_hier_{game_name}", f"pqn_hier_llm_{game_name}", f"pqn_hier_comb_{game_name}"]
    else:
        algos = [f"ppo_{game_name}", f"pqn_{game_name}", f"pqn_hier_baseline_{game_name}", f"pqn_hier_{game_name}", f"pqn_hier_llm_{game_name}", f"pqn_hier_comb_{game_name}", f"pqn_shapedreward_{game_name}", f"pqn_shapedreward_simple_{game_name}", f"pqn_shapedreward_llm_{game_name}"]
    for i, opt in enumerate(opts):
        single_reward_plot(df, algos, opt, axes[i], opt_names[i])
    # rightmost column: environment return curve for the same algorithms
    if len(axes) > len(opts):
        env_ax = axes[len(opts)]
        single_reward_plot(df, algos, "env", env_ax, "Environment Return")
        env_ax.set_ylabel("Environment Return")


with plt.rc_context({
    **bundles.neurips2024(usetex=False),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "font.size": 12,
}):
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, 4, figsize=(20, 6))
    seaquest_opt_names = ["Rescue Divers", "Shoot Enemies", "Surface"]
    kangaroo_opt_names = ["Move Up", "Handle Threats", "Collect Fruits"]
    multi_reward_plot(seaquest_df, seaquest_opt_names, axs[0])
    multi_reward_plot(kangaroo_df, kangaroo_opt_names, axs[1])

    # Create shared legend
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.11))
    # One y-label for the first row (Seaquest)
    fig.text(0.005, 0.77, "Skill Returns", va="center", ha="center", rotation="vertical", fontsize=15)
    # One y-label for the second row (Kangaroo)
    fig.text(0.005, 0.27, "Skill Returns", va="center", ha="center", rotation="vertical", fontsize=15)

    fig.text(-0.02, 0.77, "Seaquest", va="center", ha="center", rotation="vertical", fontsize=15, fontweight="bold")
    fig.text(-0.02, 0.27, "Kangaroo", va="center", ha="center", rotation="vertical", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "new_reward_plot_new_runs.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "new_reward_plot_new_runs.svg", bbox_inches="tight")
    plt.close(fig)
