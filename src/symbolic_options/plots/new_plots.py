"""
Bar-plot script comparing methods on:
  - original env return (test_clean/returned_episode_env_returns)
  - aligned metric (Seaquest → total_divers_collected, Kangaroo → game_progress)

Usage:
    uv run python src/symbolic_options/plots/new_plots.py \\
        --data-dir src/symbolic_options/plots/wandb_data/nexus_noisy \\
        [--best] [--env Seaquest] [--output my_plot.pdf]

Iterates over a wandb data directory, aggregates across rng seeds within each
run, groups runs by method, and produces one bar plot per game.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Which metric to use for the original environment return.
ENV_RETURN_METRIC = "returned_episode_env_returns"

# Aligned (symbolic / interpretable) metrics per game.
ALIGNED_METRICS = {
    "Seaquest": "total_divers_collected",
    "Kangaroo": "game_progress",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng_seed_names(history: pd.DataFrame) -> list[str]:
    """Return sorted list of unique rng seed prefixes found in *history*."""
    seeds: set[str] = set()
    for col in history.columns:
        if col.startswith("rng"):
            seeds.add(col.split("/")[0])
    return sorted(seeds)


def _metric_column(prefix: str, seed: str | None, metric: str) -> str:
    """Build the full column name for a metric, optionally per-seed.

    prefix: 'test_clean' for test eval, '' for training.
    """
    base = f"{prefix}/{metric}" if prefix else metric
    return f"{seed}/{base}" if seed else base


def _extract_value(
    history: pd.DataFrame,
    col: str,
    mode: str,  # "latest" | "best"
    lower_is_better: bool = False,
    smooth: int = 0,
) -> float:
    """Return a scalar from *col* according to *mode*.

    When mode='latest' and smooth>0, averages over the last *smooth* steps.
    """
    series = history[col].dropna()
    if series.empty:
        return np.nan
    if mode == "best":
        return float(series.min() if lower_is_better else series.max())
    # "latest"
    if smooth > 1 and len(series) >= smooth:
        return float(series.iloc[-smooth:].median())
    return float(series.iloc[-1])


def extract_method(run_name: str) -> str:
    """Strip the trailing '_EnvName' and '_noisy0' to recover the method identifier.

    Examples
    --------
    'pqn_Seaquest'                -> 'pqn'
    'pqn_hier_llm_Kangaroo'       -> 'pqn_hier_llm'
    'pqn_noisy0_Kangaroo'         -> 'pqn'
    'pqn_hier_baseline_noisy0_Seaquest' -> 'pqn_hier_baseline'
    """
    import re
    # Strip env name suffix
    parts = run_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("Seaquest", "Kangaroo"):
        run_name = parts[0]
    # Strip _noisy0 (or _noisy followed by digits)
    run_name = re.sub(r"_noisy\d*$", "", run_name)
    return run_name


def _safe_mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and std of *values*, returning (nan, 0) for empty lists."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


# ---------------------------------------------------------------------------
# Core data loading
# ---------------------------------------------------------------------------


def load_runs(data_dir: Path) -> pd.DataFrame:
    """Load all runs from *data_dir* and return a flat DataFrame.

    Each row = one run with columns:
        run_id, run_name, env_name, method, noisy_training,
        env_return_mean, env_return_std, aligned_mean, aligned_std,
        aligned_metric
    """
    rows: list[dict] = []

    for project_dir in data_dir.iterdir():
        if not project_dir.is_dir():
            continue

        for run_dir in project_dir.iterdir():
            if not run_dir.is_dir():
                continue

            config_path = run_dir / "config.json"
            history_path = run_dir / "history.parquet"
            meta_path = run_dir / "metadata.json"

            if not config_path.exists() or not history_path.exists():
                continue

            config = json.loads(config_path.read_text())
            history = pd.read_parquet(history_path)

            alg = config.get("alg", {})
            env_name = alg.get("ENV_NAME", "")
            noisy_training = alg.get("NOISY_TRAINING", False)
            # Treat None as False
            if noisy_training is None:
                noisy_training = False

            # Run name from metadata (fall back to run_id)
            run_name = run_dir.name
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                run_name = meta.get("name", run_name)

            method = extract_method(run_name)

            aligned_metric = ALIGNED_METRICS.get(env_name)

            rows.append({
                "run_id": run_dir.name,
                "run_name": run_name,
                "env_name": env_name,
                "method": method,
                "noisy_training": noisy_training,
                "history": history,
                "aligned_metric": aligned_metric,
            })

    return pd.DataFrame(rows)


def compute_run_metrics(
    df: pd.DataFrame,
    mode: str = "latest",
    prefix: str = "test_clean",
    smooth: int = 0,
) -> pd.DataFrame:
    """For each run, compute mean/std across rng seeds for both metrics.

    prefix: 'test_clean' for test eval, '' for training metrics.
    smooth: when >1 and mode='latest', average over the last *smooth* steps.
    """
    results = []

    for _, row in df.iterrows():
        history: pd.DataFrame = row["history"]
        seeds = _rng_seed_names(history)
        aligned_metric = row["aligned_metric"]

        env_returns: list[float] = []
        aligned_vals: list[float] = []

        if seeds:
            for seed in seeds:
                col = _metric_column(prefix, seed, ENV_RETURN_METRIC)
                if col in history.columns:
                    env_returns.append(_extract_value(history, col, mode, smooth=smooth))
                if aligned_metric:
                    col_a = _metric_column(prefix, seed, aligned_metric)
                    if col_a in history.columns:
                        aligned_vals.append(
                            _extract_value(history, col_a, mode, smooth=smooth)
                        )
        else:
            # Fallback: use aggregated (non-seed) column
            col = _metric_column(prefix, None, ENV_RETURN_METRIC)
            if col in history.columns:
                env_returns.append(_extract_value(history, col, mode, smooth=smooth))
            if aligned_metric:
                col_a = _metric_column(prefix, None, aligned_metric)
                if col_a in history.columns:
                    aligned_vals.append(
                        _extract_value(history, col_a, mode, smooth=smooth)
                    )

        env_mean, env_std = _safe_mean_std(env_returns)
        aligned_mean, aligned_std = _safe_mean_std(aligned_vals)

        def _count_valid(vals: list[float]) -> int:
            return int(np.sum(~np.isnan(np.asarray(vals, dtype=float))))

        results.append({
            "run_id": row["run_id"],
            "run_name": row["run_name"],
            "env_name": row["env_name"],
            "method": row["method"],
            "noisy_training": row["noisy_training"],
            "aligned_metric": aligned_metric,
            "env_return_mean": env_mean,
            "env_return_std": env_std,
            "env_n_seeds": _count_valid(env_returns),
            "aligned_mean": aligned_mean,
            "aligned_std": aligned_std,
            "aligned_n_seeds": _count_valid(aligned_vals),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Aggregation & normalisation
# ---------------------------------------------------------------------------


def aggregate_by_method(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Group by game + method, averaging across runs.

    Propagates seed-level std: when n=1 run, uses that run's seed std;
    when n>1, combines across-run std with within-run seed std.

    Returns a DataFrame with columns:
        env_name, method, aligned_metric,
        env_return_mean, env_return_std,
        aligned_mean, aligned_std,
        n_runs
    """
    grouped = metrics_df.groupby(["env_name", "method", "aligned_metric"])

    agg_rows: list[dict] = []
    for (env_name, method, aligned_metric), grp in grouped:
        row = {
            "env_name": env_name,
            "method": method,
            "aligned_metric": aligned_metric,
            "n_runs": int(grp["run_id"].count()),
        }
        # Aggregate over ALL per-seed values directly (each rng* column is one
        # independent seeded rerun). mean/std therefore equal np.mean/np.std
        # (ddof=1) computed on the flattened set of per-seed values.
        for mean_col, std_col, n_col, out_mean, out_std, out_n in (
            ("env_return_mean", "env_return_std", "env_n_seeds",
             "env_return_mean", "env_return_std", "env_n_seeds"),
            ("aligned_mean", "aligned_std", "aligned_n_seeds",
             "aligned_mean", "aligned_std", "aligned_n_seeds"),
        ):
            ok = grp[mean_col].notna()
            n = int(grp.loc[ok, n_col].sum())  # total number of seed values
            row[out_n] = n
            if n == 0:
                row[out_mean] = np.nan
                row[out_std] = 0.0
                continue
            # Grand mean = seed-count-weighted mean of per-run means.
            grand = float(np.average(grp.loc[ok, mean_col], weights=grp.loc[ok, n_col]))
            row[out_mean] = grand
            if n > 1:
                # Exact sample std over all seed values (df=1) via the ANOVA
                # identity: total_SS = within_SS + between_SS, with
                #   within_SS  = Σ_g (n_g - 1)·s_g²
                #   between_SS = Σ_g n_g·(x̄_g - x̄)²
                # and total variance = total_SS / (N - 1).
                within_ss = float(np.sum(
                    (grp.loc[ok, n_col].values - 1) * grp.loc[ok, std_col].values ** 2
                ))
                between_ss = float(np.sum(
                    grp.loc[ok, n_col].values * (grp.loc[ok, mean_col].values - grand) ** 2
                ))
                row[out_std] = float(np.sqrt((within_ss + between_ss) / (n - 1)))
            else:
                row[out_std] = 0.0
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_game(agg: pd.DataFrame, game: str, output_path: Path | None = None):
    """Create a grouped bar plot for a single game with dual y-axes.

    Left axis: env return.  Right axis: aligned metric.
    """
    game_data = agg[agg["env_name"] == game].copy()
    if game_data.empty:
        print(f"No data for {game}, skipping.")
        return

    methods = game_data["method"].tolist()
    x = np.arange(len(methods))
    width = 0.35

    env_means = game_data["env_return_mean"].values
    env_stds = game_data["env_return_std"].values
    aligned_means = game_data["aligned_mean"].values
    aligned_stds = game_data["aligned_std"].values

    # Detect if aligned metric was normalised
    aligned_label = game_data["aligned_metric"].iloc[0]

    fig, ax1 = plt.subplots(figsize=(max(8, len(methods) * 1.2), 5))
    ax2 = ax1.twinx()

    # --- Left axis: env return ---
    color1 = sns.color_palette("Set2")[0]
    bars1 = ax1.bar(
        x - width / 2,
        env_means,
        width,
        yerr=env_stds,
        label="Env Return",
        color=color1,
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Env Return", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # --- Right axis: aligned metric ---
    color2 = sns.color_palette("Set2")[1]
    bars2 = ax2.bar(
        x + width / 2,
        aligned_means,
        width,
        yerr=aligned_stds,
        label=aligned_label,
        color=color2,
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel(aligned_label, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # --- Value labels on bars ---
    for bar in bars1:
        h = bar.get_height()
        if not np.isnan(h):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color1,
            )
    for bar in bars2:
        h = bar.get_height()
        if not np.isnan(h):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=color2,
            )

    # --- Shared decorations ---
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    ax1.set_xlabel("Method")
    ax1.set_title(f"{game} — Method Comparison", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right", fontsize=8)

    # n_seeds annotation below each method (number of seeded reruns aggregated)
    for i, n in enumerate(game_data["env_n_seeds"]):
        ax1.text(
            i,
            ax1.get_ylim()[0] - 0.02 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
            f"n={n}",
            ha="center",
            va="top",
            fontsize=7,
            color="grey",
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Noisy comparison plotting
# ---------------------------------------------------------------------------


def plot_noisy_comparison(
    base_agg: pd.DataFrame,
    noisy_agg: pd.DataFrame,
    game: str,
    metric: str,  # "env" | "progress"
    noisy_label: str,
    output_path: Path | None = None,
):
    """Bar plot comparing base vs noisy for a single metric and game."""
    base_data = base_agg[base_agg["env_name"] == game].copy()
    noisy_data = noisy_agg[noisy_agg["env_name"] == game].copy()

    if base_data.empty and noisy_data.empty:
        print(f"No data for {game}, skipping.")
        return

    # Align methods across both groups
    all_methods = sorted(
        set(base_data["method"].tolist()) | set(noisy_data["method"].tolist())
    )
    x = np.arange(len(all_methods))
    width = 0.35

    def _get_vals(data, methods):
        means = []
        stds = []
        for m in methods:
            row = data[data["method"] == m]
            if len(row) > 0:
                r = row.iloc[0]
                if metric == "env":
                    means.append(r["env_return_mean"])
                    stds.append(r["env_return_std"])
                else:
                    means.append(r["aligned_mean"])
                    stds.append(r["aligned_std"])
            else:
                means.append(np.nan)
                stds.append(0.0)
        return np.array(means, dtype=float), np.array(stds, dtype=float)

    base_means, base_stds = _get_vals(base_data, all_methods)
    noisy_means, noisy_stds = _get_vals(noisy_data, all_methods)

    # Metric label
    if metric == "env":
        ylabel = "Env Return"
    else:
        ylabel = base_data["aligned_metric"].iloc[0] if len(base_data) > 0 else noisy_data["aligned_metric"].iloc[0]

    fig, ax = plt.subplots(figsize=(max(8, len(all_methods) * 1.2), 5))

    colors = sns.color_palette("Set2")
    ax.bar(
        x - width / 2, base_means, width, yerr=base_stds,
        label="Base (clean eval)", color=colors[0],
        capsize=5, edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, noisy_means, width, yerr=noisy_stds,
        label=noisy_label, color=colors[1],
        capsize=5, edgecolor="black", linewidth=0.5,
    )

    # Value labels
    for i, (b, n) in enumerate(zip(base_means, noisy_means)):
        if not np.isnan(b):
            ax.text(i - width / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=7)
        if not np.isnan(n):
            ax.text(i + width / 2, n, f"{n:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("Method")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{game} — Base vs {noisy_label} ({ylabel})", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, rotation=25, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bar-plot comparison of env return and aligned metrics."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing wandb run data (e.g. wandb_data/nexus_noisy).",
    )
    p.add_argument(
        "--best",
        action="store_true",
        default=False,
        help="Use the best (peak) value across training history instead of the latest (final) value.",
    )
    p.add_argument(
        "--training",
        action="store_true",
        default=False,
        help="Use training metrics instead of test_clean evaluation metrics.",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="N",
        help="Smooth the 'latest' value by averaging over the last N steps (ignored in --best mode).",
    )
    p.add_argument(
        "--env",
        type=str,
        default=None,
        help="Only plot a specific game (Seaquest or Kangaroo). If omitted, plots both.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save the plot. If omitted, the plot is displayed interactively.",
    )
    p.add_argument(
        "--noisy",
        type=str,
        choices=["eval", "train"],
        default=None,
        help="Compare base (clean eval, non-noisy training) against noisy eval results. "
             "'eval': noisy eval from non-noisy-trained runs. "
             "'train': noisy eval from noisy-trained runs.",
    )
    p.add_argument(
        "--mod",
        action="store_true",
        default=False,
        help="Compare base (clean eval, non-noisy training) against mod (no-enemies) eval.",
    )
    p.add_argument(
        "--metric",
        type=str,
        choices=["env", "progress"],
        default=None,
        help="Which metric to plot in noisy-comparison mode. "
             "'env': env return. 'progress': aligned metric. Ignored without --noisy.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    mode = "best" if args.best else "latest"
    prefix = "" if args.training else "test_clean"
    smooth = args.smooth if mode == "latest" else 0

    # 1. Load raw data
    print(f"Loading runs from {data_dir} ...")
    raw = load_runs(data_dir)
    print(f"  Found {len(raw)} runs.")

    if args.noisy or args.mod:
        # --- Comparison mode (noisy or mod) ---
        clean = raw[~raw["noisy_training"].astype(bool)].copy()
        base_metrics = compute_run_metrics(clean, mode=mode, prefix="test_clean", smooth=smooth)
        base_agg = aggregate_by_method(base_metrics)

        if args.mod:
            comp_prefix = "test_mod"
            comp_label = "Mod eval"
            comp_metrics = compute_run_metrics(clean, mode=mode, prefix=comp_prefix, smooth=smooth)
        elif args.noisy == "eval":
            comp_prefix = "test_noisy"
            comp_label = "Noisy eval"
            comp_metrics = compute_run_metrics(clean, mode=mode, prefix=comp_prefix, smooth=smooth)
        else:  # args.noisy == "train"
            comp_prefix = "test_noisy"
            comp_label = "Noisy eval (noisy trained)"
            noisy_runs = raw[raw["noisy_training"].astype(bool)].copy()
            if noisy_runs.empty:
                print("No noisy-training runs found. Exiting.")
                sys.exit(0)
            comp_metrics = compute_run_metrics(noisy_runs, mode=mode, prefix=comp_prefix, smooth=smooth)

        comp_agg = aggregate_by_method(comp_metrics)
        metric = args.metric or "env"

        games = sorted(set(base_agg["env_name"].unique()) | set(comp_agg["env_name"].unique()))
        if args.env:
            games = [g for g in games if g == args.env]

        sns.set_theme(style="white", context="notebook")
        for game in games:
            output_path = None
            if args.output and len(games) == 1:
                output_path = args.output
            elif args.output:
                stem = args.output.stem
                suffix = args.output.suffix
                output_path = args.output.with_name(f"{stem}_{game}{suffix}")
            print(f"Plotting {game} (base vs {comp_label}, metric={metric}) ...")
            plot_noisy_comparison(
                base_agg, comp_agg, game, metric=metric,
                noisy_label=comp_label, output_path=output_path,
            )
        return

    # --- Normal dual-axis mode ---
    # 2. Filter out noisy-training runs
    noisy_mask = raw["noisy_training"].astype(bool)
    clean = raw[~noisy_mask].copy()
    n_skipped = noisy_mask.sum()
    print(f"  Kept {len(clean)} runs (skipped {n_skipped} with noisy_training=True).")

    if clean.empty:
        print("No runs left after filtering. Exiting.")
        sys.exit(0)

    # 3. Compute per-run metrics (aggregate across rng seeds)
    print(f"Computing per-run metrics (mode={mode}, prefix='{prefix or '(training)'}', smooth={smooth}) ...")
    metrics = compute_run_metrics(clean, mode=mode, prefix=prefix, smooth=smooth)

    # 4. Aggregate by method
    agg = aggregate_by_method(metrics)

    # 5. Determine which games to plot
    games = sorted(agg["env_name"].unique())
    if args.env:
        games = [g for g in games if g == args.env]

    # 5. Plot
    sns.set_theme(style="white", context="notebook")

    for game in games:
        output_path = None
        if args.output and len(games) == 1:
            output_path = args.output
        elif args.output:
            # Multiple games → append game name to output path
            stem = args.output.stem
            suffix = args.output.suffix
            output_path = args.output.with_name(f"{stem}_{game}{suffix}")

        print(f"Plotting {game} ...")
        plot_game(agg, game, output_path=output_path)


if __name__ == "__main__":
    main()
