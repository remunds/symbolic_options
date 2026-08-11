"""
Old-style rendering of the new_plots.py data pipeline.

Everything data-related (loading runs, per-run seed aggregation, per-method
aggregation) is imported from new_plots.py and computed exactly as there.
Only the *figure design* follows plots.ipynb: tueplots (neurips2024 bundle +
axes.lines rc-context), seaborn white style (no gridlines), colorblind/pastel palettes,
twin-axis bar plots with bold colored y-labels, rotated tick labels, and
hatched 'noisy' bars with Patch legends.

Usage:
    python src/symbolic_options/plots/plots_old_style.py \\
        --data-dir src/symbolic_options/plots/wandb_data/nexus_noisy_fix_scaled \\
        [--noisy train|eval] [--best] [--smooth N] [--prefix old_style]

Produces:
    <prefix>_comparison_dual_metric.{pdf,svg}          (one fig per game, 1xN)
    <prefix>_noisy_comparison_dual_metric.{pdf,svg}    (original vs noisy eval)
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from tueplots import axes, bundles

sys.path.insert(0, str(Path(__file__).parent))
import new_plots as np_  # noqa: E402

plt.rcParams.update({"figure.dpi": 150})  # as in plots.ipynb cell 10

# ---------------------------------------------------------------------------
# Old-style (plots.ipynb) figure design
# ---------------------------------------------------------------------------

OLD_STYLE_RC = {
    **bundles.neurips2024(usetex=False),
    **axes.lines(),
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 12,
}

# Notebook cell 2: method -> display name (order matters).
def get_algorithm_from_name(name: str) -> str:
    if "ppo" in name:
        return "PPO (baseline)"
    elif "baseline" in name:
        return "HPQN (baseline)"
    elif "pre" in name:
        return "Pretrained"
    elif "comb" in name:
        return "NEXUS (nesy)"
    elif "llm" in name:
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


# Notebook cell 7: Seaquest's comb runs were named '*_shootdefault'; the
# notebook stripped that suffix so they merge into plain 'pqn_hier_comb'.
METHOD_ALIASES = {"pqn_hier_comb_shootdefault": "pqn_hier_comb"}

# Notebook cell 16 order (without PPO, which is absent in this dataset).
METHOD_ORDER = ["pqn", "pqn_hier_baseline", "pqn_hier", "pqn_hier_llm", "pqn_hier_comb"]

GAME_ORDER = ["Seaquest", "Kangaroo"]

# Right-axis labels for the aligned metrics (as aggregated by new_plots.py).
ALIGNED_LABELS = {
    "Seaquest": "Divers Rescued",
    "Kangaroo": "Game Progress",
}


def old_style_palettes():
    """colorblind + pastel palettes, as used throughout plots.ipynb."""
    return sns.color_palette("colorblind"), sns.color_palette("pastel")


def bar_colors_for(algos: list[str]) -> tuple[list, list]:
    """Baselines -> pastel shades, NEXUS variants -> colorblind shades."""
    colors, pastels = old_style_palettes()
    baselines = {"ppo", "pqn", "pqn_hier_baseline", "nudge", "blendrl"}
    hns, comp = [], []
    for algo in algos:
        if algo in baselines:
            hns.append(pastels[0])
            comp.append(pastels[1])
        else:
            hns.append(colors[0])
            comp.append(colors[1])
    return hns, comp


def methods_for_game(agg: pd.DataFrame, game: str) -> list[str]:
    """Ordered method list present for *game*."""
    present = set(agg.loc[agg["env_name"] == game, "method"])
    return [a for a in METHOD_ORDER if a in present]


def get_vals(agg: pd.DataFrame, game: str, method: str, metric: str) -> tuple[float, float]:
    """(mean, std) from the aggregated df; (nan, 0) if absent."""
    row = agg[(agg["env_name"] == game) & (agg["method"] == method)]
    if len(row) == 0:
        return np.nan, 0.0
    r = row.iloc[0]
    if metric == "env":
        return r["env_return_mean"], r["env_return_std"]
    return r["aligned_mean"], r["aligned_std"]


def rotate_xticklabels(ax, rotation=30):
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")


# ---------------------------------------------------------------------------
# Plotting (design ported from plots.ipynb cells 15 / 23)
# ---------------------------------------------------------------------------


def dual_metric_plot(agg: pd.DataFrame, game: str, algos: list[str], ax):
    """Grouped twin-axis bar plot: Game Reward | aligned metric.

    Ported from plots.ipynb cell 15 (dual_metric_plot), fed by new_plots.py
    aggregation instead of per-seed arrays.
    """
    colors, _ = old_style_palettes()
    hns_colors, comp_colors = bar_colors_for(algos)
    aligned_label = ALIGNED_LABELS.get(game, "Aligned metric")

    env_means = [get_vals(agg, game, a, "env")[0] for a in algos]
    env_stds = [get_vals(agg, game, a, "env")[1] for a in algos]
    al_means = [get_vals(agg, game, a, "aligned")[0] for a in algos]
    al_stds = [get_vals(agg, game, a, "aligned")[1] for a in algos]

    x = np.arange(len(algos))
    ax2 = ax.twinx()

    ax.bar(x - 0.2, env_means, yerr=env_stds, width=0.4, label="Game Reward", color=hns_colors)
    ax.set_ylabel("Game Reward", color=colors[0], fontweight="bold")
    ax.tick_params(axis="y", labelcolor=colors[0])

    ax2.bar(x + 0.2, al_means, yerr=al_stds, width=0.4, label=aligned_label, color=comp_colors)
    ax2.set_ylabel(aligned_label, color=colors[1], fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels([get_algorithm_from_name(a) for a in algos])
    rotate_xticklabels(ax)

    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.set_title(game)


def noisy_dual_metric_plot(base_agg, noisy_agg, game: str, algos: list[str], ax) -> list[Patch]:
    """Original (clean eval) vs Noisy eval, interleaved + hatched.

    Ported from plots.ipynb cell 23 (noisy_dual_metric_plot): for each method
    four bars at one x-position: orig-env, noisy-env, orig-aligned, noisy-aligned.
    Returns the Patch legend elements.
    """
    colors, _ = old_style_palettes()
    aligned_label = ALIGNED_LABELS.get(game, "Aligned metric")

    orig_env = [get_vals(base_agg, game, a, "env")[0] for a in algos]
    orig_env_std = [get_vals(base_agg, game, a, "env")[1] for a in algos]
    noisy_env = [get_vals(noisy_agg, game, a, "env")[0] for a in algos]
    noisy_env_std = [get_vals(noisy_agg, game, a, "env")[1] for a in algos]
    orig_al = [get_vals(base_agg, game, a, "aligned")[0] for a in algos]
    orig_al_std = [get_vals(base_agg, game, a, "aligned")[1] for a in algos]
    noisy_al = [get_vals(noisy_agg, game, a, "aligned")[0] for a in algos]
    noisy_al_std = [get_vals(noisy_agg, game, a, "aligned")[1] for a in algos]

    x = np.arange(len(algos))
    ax2 = ax.twinx()

    ax.bar(x - 0.3, orig_env, yerr=orig_env_std, width=0.2, label="Original",
           color=colors[0], edgecolor="black")
    ax.bar(x - 0.1, noisy_env, yerr=noisy_env_std, width=0.2, label="Noisy",
           color=colors[0], hatch="///", edgecolor="black")
    ax.set_ylabel("Game Reward", color=colors[0], fontweight="bold")
    ax.tick_params(axis="y", labelcolor=colors[0])

    ax2.bar(x + 0.1, orig_al, yerr=orig_al_std, width=0.2,
            label="Original", color=colors[1], edgecolor="black")
    ax2.bar(x + 0.3, noisy_al, yerr=noisy_al_std, width=0.2,
            label="Noisy", color=colors[1], hatch="///", edgecolor="black")
    ax2.set_ylabel(aligned_label, color=colors[1], fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=colors[1])

    ax.set_xticks(x)
    ax.set_xticklabels([get_algorithm_from_name(a) for a in algos])
    rotate_xticklabels(ax)

    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    ax.set_title(game)

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor="black", label="Game Reward"),
        Patch(facecolor=colors[1], edgecolor="black", label="Alignment Score"),
        Patch(facecolor="white", edgecolor="black", label="Original"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Noisy"),
    ]
    return legend_elements


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Old-style (plots.ipynb) figures from the new_plots.py data pipeline."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "wandb_data" / "nexus_noisy_fix_scaled",
        help="Directory containing wandb run data (default: nexus_noisy_fix_scaled).",
    )
    p.add_argument(
        "--noisy",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Noisy comparison source: 'train' = noisy eval from noisy-trained runs "
             "(plots.ipynb semantics), 'eval' = noisy eval from clean-trained runs "
             "(new_plots --noisy eval semantics).",
    )
    p.add_argument(
        "--best",
        action="store_true",
        default=False,
        help="Use the best (peak) value instead of the latest (final) value.",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="N",
        help="Smooth the 'latest' value by averaging over the last N steps.",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="old_style",
        help="Output filename prefix (default: old_style).",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default="7x3",
        metavar="WxH",
        help="Figure size in inches for the clean comparison figure "
             "(default: 7x3, as in plots.ipynb).",
    )
    p.add_argument(
        "--noisy-figsize",
        type=str,
        default="12x3",
        metavar="WxH",
        help="Figure size in inches for the original-vs-noisy figure, where bars are "
             "interleaved 4-per-method (default: 12x3 for better visibility).",
    )
    return p.parse_args()


def parse_figsize(s: str) -> tuple[float, float]:
    """Parse 'WxH' (inches) into (width, height)."""
    try:
        w, h = s.lower().split("x")
        return float(w), float(h)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"invalid figsize '{s}', expected e.g. 12x3") from None


def save_fig(fig, prefix: str, name: str):
    for ext in ("pdf", "svg"):
        out = f"{prefix}_{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved to {out}")


def tight_layout(fig):
    """Notebook calls fig.tight_layout(); silence the tueplots layout warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    mode = "best" if args.best else "latest"
    smooth = args.smooth if mode == "latest" else 0

    print(f"Loading runs from {args.data_dir} ...")
    raw = np_.load_runs(args.data_dir)
    print(f"  Found {len(raw)} runs.")

    # Notebook-compatible method naming (merge *shootdefault into pqn_hier_comb).
    raw["method"] = raw["method"].replace(METHOD_ALIASES)

    clean = raw[~raw["noisy_training"].astype(bool)].copy()
    noisy_runs = raw[raw["noisy_training"].astype(bool)].copy()

    # --- Aggregation exactly as in new_plots.py ---
    base_metrics = np_.compute_run_metrics(clean, mode=mode, prefix="test_clean", smooth=smooth)
    base_agg = np_.aggregate_by_method(base_metrics)

    if args.noisy == "train":
        comp_metrics = np_.compute_run_metrics(noisy_runs, mode=mode, prefix="test_noisy", smooth=smooth)
        noisy_label = "noisy eval (noisy trained)"
    else:
        comp_metrics = np_.compute_run_metrics(clean, mode=mode, prefix="test_noisy", smooth=smooth)
        noisy_label = "noisy eval (clean trained)"
    comp_agg = np_.aggregate_by_method(comp_metrics)

    games = [g for g in GAME_ORDER if g in set(base_agg["env_name"])]

    # --- Figure 1: clean comparison (notebook cell 16) ---
    with plt.rc_context(OLD_STYLE_RC):
        sns.set_style("white")
        fig, axs = plt.subplots(1, len(games), figsize=parse_figsize(args.figsize))
        if len(games) == 1:
            axs = [axs]
        for ax, game in zip(axs, games):
            dual_metric_plot(base_agg, game, methods_for_game(base_agg, game), ax)
        tight_layout(fig)
        save_fig(fig, args.prefix, "comparison_dual_metric")
        plt.close(fig)

    # --- Figure 2: original vs noisy (notebook cell 24) ---
    with plt.rc_context(OLD_STYLE_RC):
        sns.set_style("white")
        fig, axs = plt.subplots(1, len(games), figsize=parse_figsize(args.noisy_figsize))
        if len(games) == 1:
            axs = [axs]
        legend_elements = None
        for ax, game in zip(axs, games):
            legend_elements = noisy_dual_metric_plot(
                base_agg, comp_agg, game, methods_for_game(comp_agg, game), ax
            )
        fig.legend(handles=legend_elements, loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, -0.12))
        tight_layout(fig)
        save_fig(fig, args.prefix, f"noisy_comparison_dual_metric")
        plt.close(fig)

    print(f"\nDone. Figures use {noisy_label} for the noisy bars.")


if __name__ == "__main__":
    main()
