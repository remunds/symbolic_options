"""
Drop plot: original (clean) vs modified-environment performance.

Games:
  * Seaquest, Kangaroo -- NEW data from NEXUS_noisy_fix_scaled
    (NOISY_TRAINING=False). Original = test_clean aligned metric, Modified =
    test_mod aligned metric, both aggregated across ALL seeds (mean/std over
    the 6 seeded reruns) exactly as in new_plots.py.
  * Crafter -- SAME data as plots.ipynb (raban-emunds-tu-darmstadt/LENS, rnn
    craftax runs; test/ vs test_modif/ env returns, HNS-normalized with
    random=0.0, human=14.3 reward). The 3 runs per algorithm are concatenated
    along the seed axis as in notebook cell 9 (3 runs x 3 seeds = 9 seeds).

Figure design from plots.ipynb cell 20 (aligned_performance_drop_plot):
tueplots neurips2024 + axes.lines(), seaborn white style (no gridlines), pastel/colorblind
bars, rotated tick labels, Original/Modified legend.

Usage:
    python src/symbolic_options/plots/plots_drop_plot.py \\
        --data-dir src/symbolic_options/plots/wandb_data/nexus_noisy_fix_scaled \\
        [--best] [--prefix drop_plot]
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
    GAME_ORDER,
    METHOD_ALIASES,
    METHOD_ORDER,
    get_algorithm_from_name,
    get_vals,
    tight_layout,
)

plt.rcParams.update({"figure.dpi": 150})

# ---------------------------------------------------------------------------
# Crafter data (same runs as plots.ipynb cell 5: raban-emunds-tu-darmstadt/LENS)
# ---------------------------------------------------------------------------

CRAFTAX_RUNS = {  # 3 seeds per algorithm
    "pqn_rnn": ["wp2vauwf", "czpg2pbm", "ahfl9eck"],
    "pqn_rnn_hier": ["b2xbbnmj", "ft94vci9", "nu54qqys"],
    "pqn_rnn_hier_baseline": ["af0jc5od", "ecfgbptp", "kptt5kp4"],
    "pqn_rnn_hier_llm": ["zekuwdpv", "tp0291gp", "g3dk9okc"],
    "pqn_rnn_hier_comb": ["niloepr4", "qqpmrecj", "4ozz351u"],
}

# Notebook cell 20 drop plot only shows these four (no baseline).
CRAFTAX_ALGOS = ["pqn_rnn", "pqn_rnn_hier", "pqn_rnn_hier_llm", "pqn_rnn_hier_comb"]

LENS_PROJECT = "raban-emunds-tu-darmstadt/LENS"

# (random, human) env-return for HNS normalisation (notebook cell 11).
HUMAN_RANDOM = {"craftax-classic-symbolic-v1": (0.0, 14.3)}

DROP_RC = {
    **bundles.neurips2024(usetex=False),
    **axes.lines(),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.size": 12,
}


# ---------------------------------------------------------------------------
# Crafter loading / HNS (faithful port of notebook cells 3 + 9 + 20)
# ---------------------------------------------------------------------------


def get_wandb_api():
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not found (checked env + .env).", file=sys.stderr)
        sys.exit(1)
    import wandb

    return wandb.Api()


def load_craftax_history(api, cache_dir: Path, run_hash: str) -> pd.DataFrame:
    """History for one crafter run, cached locally as parquet."""
    p = cache_dir / f"{run_hash}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    h = api.run(f"{LENS_PROJECT}/runs/{run_hash}").history()
    p.parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(p)
    print(f"  cached crafter run {run_hash} -> {p}")
    return h


def craftax_drop_values(api, cache_dir: Path, algo: str) -> tuple:
    """Original/modified HNS + stds for one crafter algorithm.

    Concatenates seed arrays of the algorithm's 3 runs (as notebook cell 9),
    i.e. 9 seeds, then computes HNS at the argmax timestep of the mean curve.
    """
    clean_col = "/test/returned_episode_env_returns"
    mod_col = "/test_modif/returned_episode_env_returns"
    clean_arrs, mod_arrs, n_seeds = [], [], 0
    for run_hash in CRAFTAX_RUNS[algo]:
        h = load_craftax_history(api, cache_dir, run_hash)
        seeds = sorted({c.split("/")[0] for c in h.columns if c.startswith("rng")})
        for s in seeds:
            c, m = f"{s}{clean_col}", f"{s}{mod_col}"
            if c in h.columns and m in h.columns:
                clean_arrs.append(h[c].to_numpy(dtype=float))
                mod_arrs.append(h[m].to_numpy(dtype=float))
                n_seeds += 1
    if n_seeds == 0:
        raise RuntimeError(f"no test/test_modif seed columns found for {algo}")
    lens = {len(a) for a in clean_arrs} | {len(a) for a in mod_arrs}
    assert len(lens) == 1, f"seed arrays have mismatched lengths for {algo}: {lens}"

    arr = np.array(clean_arrs)  # (n_seeds, n_steps)
    arr_m = np.array(mod_arrs)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean_m = np.nanmean(arr_m, axis=0)
    std_m = np.nanstd(arr_m, axis=0)
    max_idx = int(np.nanargmax(mean))

    random_score, human_score = HUMAN_RANDOM["craftax-classic-symbolic-v1"]
    denom = human_score - random_score + 1e-8
    hns = (mean[max_idx] - random_score) / denom
    hns_std = std[max_idx] / denom
    hns_mod = (mean_m[max_idx] - random_score) / denom
    hns_mod_std = std_m[max_idx] / denom
    print(f"  {algo:22s} n_seeds={n_seeds} hns={hns:.3f} (+/- {hns_std:.3f}) "
          f"mod_hns={hns_mod:.3f} (+/- {hns_mod_std:.3f})")
    return hns, hns_std, hns_mod, hns_mod_std


# ---------------------------------------------------------------------------
# Plotting (design from notebook cell 20)
# ---------------------------------------------------------------------------


def drop_plot(ax, game: str, algos: list[str], orig, orig_std, mod, mod_std, ylabel: str):
    colors = sns.color_palette("colorblind")
    pastels = sns.color_palette("pastel")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drop plot: original vs modified performance.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "wandb_data" / "nexus_noisy_fix_scaled",
        help="Local wandb data dir for Seaquest/Kangaroo (default: nexus_noisy_fix_scaled).",
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
        default="drop_plot",
        help="Output filename prefix (default: drop_plot).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    mode = "best" if args.best else "latest"
    smooth = args.smooth if mode == "latest" else 0

    # --- Seaquest / Kangaroo from the new scaled data ---
    print(f"Loading runs from {args.data_dir} ...")
    raw = np_.load_runs(args.data_dir)
    raw["method"] = raw["method"].replace(METHOD_ALIASES)
    clean = raw[~raw["noisy_training"].astype(bool)].copy()
    print(f"  {len(clean)} clean runs (NOISY_TRAINING=False).")

    base_agg = np_.aggregate_by_method(
        np_.compute_run_metrics(clean, mode=mode, prefix="test_clean", smooth=smooth)
    )
    mod_agg = np_.aggregate_by_method(
        np_.compute_run_metrics(clean, mode=mode, prefix="test_mod", smooth=smooth)
    )

    # --- Crafter from LENS (plots.ipynb data) ---
    api = get_wandb_api()
    cache_dir = Path(__file__).parent / "wandb_data" / "lens_craftax"
    print("Loading crafter data (cached under wandb_data/lens_craftax) ...")

    games = [g for g in GAME_ORDER if g in set(base_agg["env_name"])]

    with plt.rc_context(DROP_RC):
        sns.set_style("white")
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        # Seaquest + Kangaroo (new data, aligned metrics over all seeds)
        for ax, game in zip(axs[:2], games):
            algos = [a for a in METHOD_ORDER
                     if a in set(base_agg.loc[base_agg["env_name"] == game, "method"])]
            orig = [get_vals(base_agg, game, a, "aligned")[0] for a in algos]
            orig_std = [get_vals(base_agg, game, a, "aligned")[1] for a in algos]
            mod = [get_vals(mod_agg, game, a, "aligned")[0] for a in algos]
            mod_std = [get_vals(mod_agg, game, a, "aligned")[1] for a in algos]
            n_seeds = set(
                base_agg.loc[base_agg["env_name"] == game, "aligned_n_seeds"]
            )
            ylabel = "Divers Rescued" if game == "Seaquest" else "Game Progress"
            print(f"{game}: n_seeds per method = {sorted(n_seeds)}, methods = {algos}")
            drop_plot(ax, game, algos, orig, orig_std, mod, mod_std, ylabel)

        # Crafter (HNS, notebook data)
        crafter_algos = CRAFTAX_ALGOS
        hns_orig, hns_orig_std, hns_mod, hns_mod_std = [], [], [], []
        for a in crafter_algos:
            o, os_, m, ms = craftax_drop_values(api, cache_dir, a)
            hns_orig.append(o)
            hns_orig_std.append(os_)
            hns_mod.append(m)
            hns_mod_std.append(ms)
        drop_plot(axs[2], "Crafter", crafter_algos,
                  hns_orig, hns_orig_std, hns_mod, hns_mod_std, "HNS")

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
