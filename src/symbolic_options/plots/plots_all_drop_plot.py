"""
All-baselines drop plot -- recreation of the plots.ipynb 'all_drop_plot'
figure (notebook cell 31): original vs modified aligned metric for all
baselines (PPO, PQN, HPQN-baseline, NUDGE, BlendRL, NEXUS variants).

Same data convention as plots_all_baselines.py:
  * NEXUS methods: NEW data from NEXUS_noisy_fix_scaled (NOISY_TRAINING=False),
    latest value, aggregated over all 6 seeds; test_clean vs test_mod aligned
    metrics (total_divers_collected / game_progress).
  * PPO: OLD LENS data, latest over its seeds; test/ vs test_modif/ option
    returns (opt4 divers, opt3 level-completion %).
  * NUDGE / BlendRL: external scores hardcoded from notebook cell 27
    (original + modified), means unchanged.

Adaptations (same as all_baselines + summary at the end of the run):
  * Kangaroo aligned axis in '% Level Completion': NEXUS game_progress * 100,
    PPO (opt3/9)*100, NUDGE/BlendRL manual corrected %.
  * PPO Kangaroo: seed rng1797259609 excluded (notebook outlier handling).
  * Error bars: sample std (ddof=1) everywhere.

Usage:
    python src/symbolic_options/plots/plots_all_drop_plot.py \
        --data-dir src/symbolic_options/plots/wandb_data/nexus_noisy_fix_scaled \
        [--prefix all_drop_plot] [--figsize 12x3]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import new_plots as np_  # noqa: E402
from plots_all_baselines import (  # noqa: E402
    ALL_BASELINES_ORDER,
    ALL_RC,
    LENS_PROJECT,
    PPO_RUNS,
    external_values,
    get_wandb_api,
    load_ppo_history,
    nexus_values,
    ppo_values,
)
from plots_old_style import (  # noqa: E402
    METHOD_ALIASES,
    get_algorithm_from_name,
    parse_figsize,
    tight_layout,
)

plt.rcParams.update({"figure.dpi": 150})


def aligned_drop_plot(ax, game: str, algos: list[str], orig, orig_std, mod, mod_std):
    """Original vs Modified aligned-metric bars (notebook cell 20 style)."""
    colors = sns.color_palette("colorblind")
    pastels = sns.color_palette("pastel")
    ylabel = "Divers Rescued" if game == "Seaquest" else "Level Completion (%)"
    x = np.arange(len(algos))
    ax.bar(x - 0.2, orig, yerr=orig_std, width=0.4, color=pastels[1])
    ax.bar(x + 0.2, mod, yerr=mod_std, width=0.4, color=colors[1])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([get_algorithm_from_name(a) for a in algos])
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.set_ylim(bottom=0)
    ax.set_title(game)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All-baselines drop plot (Seaquest, Kangaroo).")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "wandb_data" / "nexus_noisy_fix_scaled",
        help="Local wandb data dir for the NEXUS runs.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="all_drop_plot",
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
    base_agg = np_.aggregate_by_method(
        np_.compute_run_metrics(clean, mode="latest", prefix="test_clean")
    )
    mod_agg = np_.aggregate_by_method(
        np_.compute_run_metrics(clean, mode="latest", prefix="test_mod")
    )

    # --- PPO (old LENS data) ---
    api = get_wandb_api()
    cache_dir = Path(__file__).parent / "wandb_data" / "lens_ppo"
    print("Loading old PPO data (cached under wandb_data/lens_ppo) ...")

    nexus_methods = ["pqn", "pqn_hier_baseline", "pqn_hier", "pqn_hier_llm", "pqn_hier_comb"]

    values: dict[str, dict[str, dict[str, tuple]]] = {}
    for game in ("Seaquest", "Kangaroo"):
        values[game] = {"orig": {}, "mod": {}}
        for m in nexus_methods:
            values[game]["orig"][m] = nexus_values(base_agg, game, m)
            values[game]["mod"][m] = nexus_values(mod_agg, game, m)
        values[game]["orig"]["ppo"] = ppo_values(api, cache_dir, game, mod=False)
        values[game]["mod"]["ppo"] = ppo_values(api, cache_dir, game, mod=True)
        for a in ("nudge", "blendrl"):
            values[game]["orig"][a] = external_values(game, a, mod=False)
            values[game]["mod"][a] = external_values(game, a, mod=True)

    # Only the aligned metric is shown in the drop plot (notebook cell 20).
    for game in ("Seaquest", "Kangaroo"):
        print(f"--- {game} (aligned metric, orig vs mod) ---")
        for a in ALL_BASELINES_ORDER:
            o = values[game]["orig"][a][2]
            os_ = values[game]["orig"][a][3]
            m = values[game]["mod"][a][2]
            ms = values[game]["mod"][a][3]
            print(f"  {a:20s} orig={o:9.2f} +/- {os_:6.2f}   mod={m:9.2f} +/- {ms:6.2f}")

    # --- Figure (notebook cell 31 style) ---
    with plt.rc_context(ALL_RC):
        sns.set_style("white")
        fig, axs = plt.subplots(1, 2, figsize=parse_figsize(args.figsize))
        for ax, game in zip(axs, ("Seaquest", "Kangaroo")):
            orig = [values[game]["orig"][a][2] for a in ALL_BASELINES_ORDER]
            orig_std = [values[game]["orig"][a][3] for a in ALL_BASELINES_ORDER]
            mod = [values[game]["mod"][a][2] for a in ALL_BASELINES_ORDER]
            mod_std = [values[game]["mod"][a][3] for a in ALL_BASELINES_ORDER]
            aligned_drop_plot(ax, game, ALL_BASELINES_ORDER, orig, orig_std, mod, mod_std)
        fig.legend(labels=["Original", "Modified"], loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, -0.12))
        tight_layout(fig)
        for ext in ("pdf", "svg"):
            out = f"{args.prefix}.{ext}"
            fig.savefig(out, bbox_inches="tight")
            print(f"Saved to {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
