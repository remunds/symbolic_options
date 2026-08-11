"""
All-baselines dual-metric comparison (Seaquest, Kangaroo) -- recreation of the
plots.ipynb 'all_baselines_comparison_dual_metric' figure with mixed data.

Data sources:
  * NEXUS methods (pqn, pqn_hier_baseline, pqn_hier, pqn_hier_llm,
    pqn_hier_comb): NEW data from NEXUS_noisy_fix_scaled (NOISY_TRAINING=False),
    latest value, aggregated over ALL 6 seeds via new_plots.py. Aligned metric =
    NEW logged metrics: total_divers_collected (Seaquest) and game_progress
    (Kangaroo).
  * PPO: OLD data from plots.ipynb (raban-emunds-tu-darmstadt/LENS), latest
    value over its seeds. Aligned metric = OLD option returns
    (returned_episode_returns_4 -> divers, _3 -> level-completion %).
  * NUDGE / BlendRL: external scores hardcoded from plots.ipynb cell 27
    (unchanged mean values).

Adaptations (see docstring summary at the end):
  * Kangaroo aligned axis: NEXUS game_progress (0..~2) is scaled by 100 to the
    notebook's 'Level Completion (%)' axis; PPO uses (opt3/9)*100 as in the
    notebook; NUDGE/BlendRL use the notebook's manually corrected percentages.
  * PPO Kangaroo: seed rng1797259609 excluded (weird outlier), as in the
    notebook.
  * Error bars use sample std (ddof=1) everywhere.

Usage:
    python src/symbolic_options/plots/plots_all_baselines.py \
        --data-dir src/symbolic_options/plots/wandb_data/nexus_noisy_fix_scaled \
        [--prefix all_baselines_comparison_dual_metric] [--figsize 12x3]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import axes, bundles

sys.path.insert(0, str(Path(__file__).parent))
import new_plots as np_  # noqa: E402
from plots_old_style import (  # noqa: E402
    METHOD_ALIASES,
    METHOD_ORDER,
    bar_colors_for,
    get_algorithm_from_name,
    parse_figsize,
    tight_layout,
)

plt.rcParams.update({"figure.dpi": 150})

# ---------------------------------------------------------------------------
# Old-data constants (plots.ipynb cell 5 / 27 / 30)
# ---------------------------------------------------------------------------

LENS_PROJECT = "raban-emunds-tu-darmstadt/LENS"
PPO_RUNS = {"Seaquest": "dfc6vzov", "Kangaroo": "356qhfba"}
PPO_EXCLUDED_KANGAROO_SEED = "rng1797259609"  # notebook cell 3: weird outlier

# Method order exactly as in notebook cell 30.
ALL_BASELINES_ORDER = [
    "ppo", "pqn", "pqn_hier_baseline", "nudge", "blendrl",
    "pqn_hier", "pqn_hier_llm", "pqn_hier_comb",
]

# NUDGE / BlendRL external scores (notebook cell 27).
# Each entry: (3 seeds, 3 episodes, [env_score, alignment_score]).
NUDGE_SCORES = {
    "Seaquest": [[[0, 3], [80, 6], [100, 5]],
                 [[20, 3], [60, 4], [100, 6]],
                 [[20, 3], [0, 3], [40, 4]]],
    "Kangaroo": [[[800, 4], [2300, 11], [800, 5]],
                 [[400, 4], [300, 2], [2300, 6]],
                 [[2100, 66], [2300, 63], [2400, 93]]],
}
BLENDRL_SCORES = {
    "Seaquest": [[[1820, 12], [320, 9], [1630, 9]],
                 [[360, 9], [1440, 10], [1600, 16]],
                 [[5440, 29], [5400, 28], [1240, 20]]],
    "Kangaroo": [[[2700, 95], [2200, 17], [2700, 61]],
                 [[2000, 32], [2800, 54], [2400, 93]],
                 [[300, 2], [2200, 35], [300, 3]]],
}
MOD_NUDGE_SCORES = {
    "Seaquest": [[[0, 11], [560, 14], [0, 11]],
                 [[0, 3], [0, 3], [0, 3]],
                 [[0, 3], [0, 3], [460, 15]]],
    "Kangaroo": [[[1300, 8], [1600, 251], [2000, 249]],
                 [[2300, 730], [2200, 493], [2000, 251]],
                 [[2300, 489], [1900, 490], [2100, 251]]],
}
MOD_BLENDRL_SCORES = {
    "Seaquest": [[[19060, 58], [2660, 22], [19100, 60]],
                 [[1940, 24], [85500, 144], [86980, 139]],
                 [[2880, 22], [1120, 9], [3560, 22]]],
    "Kangaroo": [[[2300, 246], [2300, 424], [2100, 194]],
                 [[1900, 244], [2000, 704], [2000, 475]],
                 [[2200, 485], [1600, 249], [1100, 489]]],
}
# Manually corrected Kangaroo level-completion % (notebook cell 27, based on
# game footage; reaching level 2 => >100%).
KANG_CORRECTED = {
    "nudge": [100, 130, 100],
    "blendrl": [105, 100, 100],
}
KANG_CORRECTED_MOD = {
    "nudge": [100, 100, 105],
    "blendrl": [150, 100, 105],
}

ALL_RC = {
    **bundles.neurips2024(usetex=False),
    **axes.lines(),
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 12,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_wandb_api():
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found (checked env + .env).", file=sys.stderr)
        sys.exit(1)
    import wandb

    return wandb.Api()


def load_ppo_history(api, cache_dir: Path, game: str) -> pd.DataFrame:
    run_hash = PPO_RUNS[game]
    p = cache_dir / f"ppo_{game}_{run_hash}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    h = api.run(f"{LENS_PROJECT}/runs/{run_hash}").history()
    p.parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(p)
    print(f"  cached PPO {game} run {run_hash} -> {p}")
    return h


def _latest_series(h: pd.DataFrame, col: str) -> float:
    if col not in h.columns:
        return np.nan
    s = h[col].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def _mean_std(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def ppo_values(api, cache_dir: Path, game: str, mod: bool = False) -> tuple[float, float, float, float]:
    """(env_mean, env_std, aligned_mean, aligned_std) for old PPO, latest.

    mod=True reads test_modif/ metrics instead of test/.
    """
    h = load_ppo_history(api, cache_dir, game)
    seeds = sorted({c.split("/")[0] for c in h.columns if c.startswith("rng")})
    if game == "Kangaroo":
        seeds = [s for s in seeds if s != PPO_EXCLUDED_KANGAROO_SEED]
    prefix = "test_modif" if mod else "test"
    opt_col = "returned_episode_returns_4" if game == "Seaquest" else "returned_episode_returns_3"
    envs, opts = [], []
    for s in seeds:
        envs.append(_latest_series(h, f"{s}/{prefix}/returned_episode_env_returns"))
        opts.append(_latest_series(h, f"{s}/{prefix}/{opt_col}"))
    env_mean, env_std = _mean_std(envs)
    opt_mean, opt_std = _mean_std(opts)
    if game == "Kangaroo":
        opt_mean, opt_std = opt_mean / 9 * 100, opt_std / 9 * 100  # -> % level completion
    return env_mean, env_std, opt_mean, opt_std


def external_values(game: str, algo: str, mod: bool = False) -> tuple[float, float, float, float]:
    """(env_mean, env_std, aligned_mean, aligned_std) for NUDGE / BlendRL.

    Port of notebook cell 27: mean over episodes (axis=1), then mean/std over
    the 3 seeds (ddof=1 here, notebook used ddof=0 -- see summary).
    """
    if mod:
        scores = (MOD_NUDGE_SCORES if algo == "nudge" else MOD_BLENDRL_SCORES)[game]
        corrected = KANG_CORRECTED_MOD[algo] if game == "Kangaroo" else None
    else:
        scores = (NUDGE_SCORES if algo == "nudge" else BLENDRL_SCORES)[game]
        corrected = KANG_CORRECTED[algo] if game == "Kangaroo" else None
    per_seed = np.mean(np.array(scores), axis=1)  # (3 seeds, 2 cols)
    env_mean, env_std = _mean_std(list(per_seed[:, 0]))
    if game == "Kangaroo":
        al_mean = float(np.mean(corrected))
        al_std = float(np.std(corrected, ddof=1))
    else:
        al_mean, al_std = _mean_std(list(per_seed[:, 1]))
    return env_mean, env_std, al_mean, al_std


def nexus_values(agg, game: str, method: str) -> tuple[float, float, float, float]:
    """(env_mean, env_std, aligned_mean, aligned_std) for new NEXUS runs."""
    row = agg[(agg["env_name"] == game) & (agg["method"] == method)]
    if len(row) == 0:
        return np.nan, 0.0, np.nan, 0.0
    r = row.iloc[0]
    aligned = r["aligned_mean"]
    aligned_std = r["aligned_std"]
    if game == "Kangaroo":
        # game_progress -> % level completion (both mean and std)
        aligned = aligned * 100.0
        aligned_std = aligned_std * 100.0
    return r["env_return_mean"], r["env_return_std"], aligned, aligned_std


# ---------------------------------------------------------------------------
# Plotting (design from notebook cells 15/30)
# ---------------------------------------------------------------------------


def dual_metric_bars(ax, game: str, algos: list[str], env_means, env_stds,
                     al_means, al_stds):
    colors = sns.color_palette("colorblind")
    hns_colors, comp_colors = bar_colors_for(algos)
    aligned_label = ("Divers Rescued" if game == "Seaquest"
                     else "Level Completion (%)")
    x = np.arange(len(algos))
    ax2 = ax.twinx()

    ax.bar(x - 0.2, env_means, yerr=env_stds, width=0.4, color=hns_colors)
    ax.set_ylabel("Game Reward", color=colors[0], fontweight="bold")
    ax.tick_params(axis="y", labelcolor=colors[0])

    ax2.bar(x + 0.2, al_means, yerr=al_stds, width=0.4, color=comp_colors)
    ax2.set_ylabel(aligned_label, color=colors[1], fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels([get_algorithm_from_name(a) for a in algos])
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.set_title(game)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="All-baselines dual-metric comparison (Seaquest, Kangaroo)."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "wandb_data" / "nexus_noisy_fix_scaled",
        help="Local wandb data dir for the NEXUS runs.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="all_baselines_comparison_dual_metric",
        help="Output filename prefix.",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default="12x3",
        metavar="WxH",
        help="Figure size in inches (default: 12x3).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    # --- NEXUS (new data, latest, 6 seeds) ---
    print(f"Loading NEXUS runs from {args.data_dir} ...")
    raw = np_.load_runs(args.data_dir)
    raw["method"] = raw["method"].replace(METHOD_ALIASES)
    clean = raw[~raw["noisy_training"].astype(bool)]
    agg = np_.aggregate_by_method(
        np_.compute_run_metrics(clean, mode="latest", prefix="test_clean")
    )

    # --- PPO (old LENS data, latest) ---
    api = get_wandb_api()
    cache_dir = Path(__file__).parent / "wandb_data" / "lens_ppo"
    print("Loading old PPO data (cached under wandb_data/lens_ppo) ...")

    values: dict[str, dict[str, tuple]] = {game: {} for game in ("Seaquest", "Kangaroo")}
    for game in ("Seaquest", "Kangaroo"):
        for method in ["pqn", "pqn_hier_baseline", "pqn_hier", "pqn_hier_llm", "pqn_hier_comb"]:
            values[game][method] = nexus_values(agg, game, method)
        values[game]["ppo"] = ppo_values(api, cache_dir, game)
        values[game]["nudge"] = external_values(game, "nudge")
        values[game]["blendrl"] = external_values(game, "blendrl")

    for game in ("Seaquest", "Kangaroo"):
        print(f"--- {game} ---")
        for a in ALL_BASELINES_ORDER:
            v = values[game][a]
            print(f"  {a:20s} env={v[0]:9.2f} +/- {v[1]:7.2f}   aligned={v[2]:8.2f} +/- {v[3]:6.2f}")

    # --- Figure ---
    with plt.rc_context(ALL_RC):
        sns.set_style("white")
        fig, axs = plt.subplots(1, 2, figsize=parse_figsize(args.figsize))
        for ax, game in zip(axs, ("Seaquest", "Kangaroo")):
            env_means = [values[game][a][0] for a in ALL_BASELINES_ORDER]
            env_stds = [values[game][a][1] for a in ALL_BASELINES_ORDER]
            al_means = [values[game][a][2] for a in ALL_BASELINES_ORDER]
            al_stds = [values[game][a][3] for a in ALL_BASELINES_ORDER]
            dual_metric_bars(ax, game, ALL_BASELINES_ORDER,
                             env_means, env_stds, al_means, al_stds)
        tight_layout(fig)
        for ext in ("pdf", "svg"):
            out = f"{args.prefix}.{ext}"
            fig.savefig(out, bbox_inches="tight")
            print(f"Saved to {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
