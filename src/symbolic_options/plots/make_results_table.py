"""
Regenerate results_table.tex with the new results.

Same table structure as plots.ipynb cell 32 / the previous results_table.tex:
4 columns per game (Default env return, aligned metric, Simplification env
return, aligned metric) x 2 games (Seaquest, Kangaroo), rows PPO, PQN,
NEXUS (neural/symbolic/nesy), NUDGE, BlendRL, SCoBots (manual, unchanged).

Data sources (same convention as plots_all_baselines / plots_all_drop_plot):
  * NEXUS: new nexus_noisy_fix_scaled data (NOISY_TRAINING=False), latest,
    6 seeds. Aligned = new metrics: total_divers_collected (Seaquest),
    game_progress x100 (Kangaroo, -> % Level Completion).
  * PPO: old LENS data, latest; aligned = opt4 (Seaquest) / (opt3/9)*100
    (Kangaroo); seed rng1797259609 excluded for Kangaroo (notebook outlier).
  * NUDGE/BlendRL: external scores from notebook cell 27 (orig + mod),
    means unchanged; Kangaroo aligned = manual corrected %.
  * Error bars: sample std (ddof=1). Bold = column max over the 7 computed
    methods (SCoBots excluded, manual).

Usage:
    python src/symbolic_options/plots/make_results_table.py \
        --data-dir src/symbolic_options/plots/wandb_data/nexus_noisy_fix_scaled
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import new_plots as np_  # noqa: E402
from plots_all_baselines import (  # noqa: E402
    external_values,
    get_wandb_api,
    nexus_values,
    ppo_values,
)
from plots_old_style import METHOD_ALIASES  # noqa: E402

# (table name, method key)
ROWS = [
    ("PPO", "ppo"),
    ("PQN", "pqn"),
    ("NEXUS (neural)", "pqn_hier"),
    ("NEXUS (symbolic)", "pqn_hier_llm"),
    ("NEXUS (nesy)", "pqn_hier_comb"),
    ("NUDGE", "nudge"),
    ("BlendRL", "blendrl"),
]

# SCoBots: manual external results, unchanged (no new data available).
SCOBOTS_ROW = (
    r"SCoBots & 1055.3 $\pm$ 272.6 & (- $\pm$ -) & 0.0 $\pm$ 0.0 & (- $\pm$ -) "
    r"& 2776.6 $\pm$ 1332.4 & (- $\pm$ -) & 0.0 $\pm$ 0.0 & (- $\pm$ -) \\"
)

TABLE_HEADER = r"""\begin{table}
\centering
\caption{Performance Comparison on Seaquest and Kangaroo}
\label{tab:performance_comp}
\resizebox{\textwidth}{!}{
\begin{tabular}{l | cc cc || cc cc}
\hline
\multirow{2}{*}{\textbf{Algorithm}} & \multicolumn{4}{c||}{\textbf{Seaquest}} & \multicolumn{4}{c}{\textbf{Kangaroo}} \\
 & Default & Divers Collected & Simplification & Divers Collected & Default & Level Completion (\%) & Simplification & Level Completion (\%) \\
\hline
"""

TABLE_FOOTER = r"""\hline
\end{tabular}
}
\end{table}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate results_table.tex.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "wandb_data" / "nexus_noisy_fix_scaled",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "results_table.tex",
        help="Output .tex file.",
    )
    return p.parse_args()


def fmt(v: float, s: float) -> str:
    return f"{v:.1f} $\\pm$ {s:.1f}"


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        print(f"Error: {args.data_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    raw = np_.load_runs(args.data_dir)
    raw["method"] = raw["method"].replace(METHOD_ALIASES)
    clean = raw[~raw["noisy_training"].astype(bool)]
    base_agg = np_.aggregate_by_method(
        np_.compute_run_metrics(clean, mode="latest", prefix="test_clean")
    )
    mod_agg = np_.aggregate_by_method(
        np_.compute_run_metrics(clean, mode="latest", prefix="test_mod")
    )
    api = get_wandb_api()
    cache_dir = Path(__file__).parent / "wandb_data" / "lens_ppo"

    # values[method] = {game: [def_env, def_env_std, def_al, def_al_std,
    #                          mod_env, mod_env_std, mod_al, mod_al_std]}
    values: dict[str, dict[str, list[float]]] = {}
    for name, key in ROWS:
        values[key] = {}
        for game in ("Seaquest", "Kangaroo"):
            if key == "ppo":
                d = ppo_values(api, cache_dir, game, mod=False)
                m = ppo_values(api, cache_dir, game, mod=True)
            elif key in ("nudge", "blendrl"):
                d = external_values(game, key, mod=False)
                m = external_values(game, key, mod=True)
            else:
                d = nexus_values(base_agg, game, key)
                m = nexus_values(mod_agg, game, key)
            values[key][game] = [d[0], d[1], d[2], d[3], m[0], m[1], m[2], m[3]]

    # Column maxima per game (over computed methods only) for bolding.
    # Per game the 4 table columns map to value indices [0, 2, 4, 6].
    best: dict[str, list[tuple[float, str]]] = {}
    for game in ("Seaquest", "Kangaroo"):
        best[game] = [(float("-inf"), None)] * 4
        for name, key in ROWS:
            v = values[key][game]
            for li, vi in enumerate([0, 2, 4, 6]):
                if v[vi] > best[game][li][0]:
                    best[game][li] = (v[vi], name)

    lines = [TABLE_HEADER]
    for name, key in ROWS:
        cell = []
        for game in ("Seaquest", "Kangaroo"):
            v = values[key][game]
            # columns: 0=def_env,1=std, 2=def_al,3=al_std, 4=mod_env,5, 6=mod_al,7
            texts = [
                fmt(v[0], v[1]),
                f"({fmt(v[2], v[3])})",
                fmt(v[4], v[5]),
                f"({fmt(v[6], v[7])})",
            ]
            for li, text in enumerate(texts):
                if best[game][li][1] == name:
                    text = f"\\textbf{{{text}}}"
                cell.append(text)
        lines.append(f"{name} & " + " & ".join(cell) + " \\\\")
    lines.append(SCOBOTS_ROW)
    lines.append(TABLE_FOOTER)

    out = "\n".join(lines)
    args.out.write_text(out)
    print(f"Wrote {args.out}")
    print(out)


if __name__ == "__main__":
    main()
