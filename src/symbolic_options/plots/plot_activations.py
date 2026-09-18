"""Compare skill-activation statistics across meta-policy ablations.

Reads the JSON files written by `eval_activations.py` and produces:

  * <prefix>_performance.pdf/svg  -- episode return and length per ablation
  * <prefix>_skill_usage.pdf/svg  -- how the option mix shifts per ablation
  * <prefix>_commitment.pdf/svg   -- option dwell length and switch rate
  * <prefix>_conditions.pdf/svg   -- rule contention, rule-following, LLM agreement
  * <prefix>_summary.csv / .md    -- the same numbers as a table

The four ablations form a ladder of how much of the meta-policy is replaced by
a random draw (see eval_activations.py):

    default   meta-policy everywhere
    tiebreak  meta-policy, except uniform over enabled skills where >1 holds
    masked    uniform over enabled skills everywhere
    uniform   uniform over all skills, ignoring the symbolic conditions

Usage:
    python src/symbolic_options/plots/plot_activations.py \
        --data-dir outputs/activations [--prefix activations]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from plots_old_style import (  # noqa: E402
    OLD_STYLE_RC,
    get_algorithm_from_name,
    parse_figsize,
)

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

# ordered from "meta-policy fully in charge" to "meta-policy fully replaced".
# The `+hold` variants hold a random draw for >= EVAL_RANDOM_HOLD steps, which
# keeps the option-commitment profile close to the real policy's so the drop in
# return is not just the loss of temporal abstraction.
MODE_ORDER = [
    "default",
    "tiebreak+hold", "tiebreak",
    "masked+hold", "masked",
    "uniform+hold", "uniform",
]
MODE_LABEL = {
    "default": "default",
    "tiebreak+hold": "tie-break\n+hold",
    "tiebreak": "tie-break",
    "masked+hold": "masked\n+hold",
    "masked": "masked",
    "uniform+hold": "uniform\n+hold",
    "uniform": "uniform",
}
GAME_ORDER = ["Seaquest", "Kangaroo"]  # narrowed to what is loaded, in main()
ALG_ORDER = ["NEXUS (symbolic)", "NEXUS (nesy)"]

# seaborn 'colorblind', validated for CVD separation and contrast
_CB = sns.color_palette("colorblind").as_hex()
ALG_COLOR = {ALG_ORDER[0]: _CB[0], ALG_ORDER[1]: _CB[3]}  # blue, vermillion
SKILL_COLORS = [_CB[0], _CB[1], _CB[2], _CB[4], _CB[9]]  # fixed order, never cycled

plt.rcParams.update({"figure.dpi": 150})

# the ablation rungs actually present in the loaded data; set once in main() so
# every panel shares one x-axis even when a cell is missing
MODES_PRESENT: list = []


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_records(data_dir: Path, seed: int) -> pd.DataFrame:
    """One row per (game, algorithm, ablation, vmap seed)."""
    rows = []
    for path in sorted(data_dir.glob("*.json")):
        with open(path) as f:
            blob = json.load(f)
        if blob.get("eval_env", "").split("_")[0] != "train":
            continue  # only the default (training) environment
        if int(blob.get("seed", -1)) != seed:
            continue  # the directory also holds ad-hoc runs of other checkpoints
        mode = blob.get("random_meta") or "default"
        hold = int(blob.get("random_hold", 1) or 1)
        if hold > 1:
            mode = f"{mode}+hold"
        alg = get_algorithm_from_name(blob["alg_name"])
        skills = blob["skill_names"]
        for vmap, res in blob["per_vmap"].items():
            n_acts = float(np.sum(res["skill_activations_per_episode_mean"]))
            rows.append(
                {
                    "game": blob["env_name"],
                    "alg": alg,
                    "alg_name": blob["alg_name"],
                    "mode": mode,
                    "seed": int(vmap),
                    "skills": tuple(skills),
                    "episode_return": res["episode_return_mean"],
                    "episode_length": res["episode_length_mean"],
                    "switches_per_ep": res["switches_per_episode_mean"],
                    # steps an option stays active once selected, across options
                    "dwell": res["episode_length_mean"] / max(n_acts, 1e-9),
                    "multi_cond": res["multi_condition_share"],
                    "rule_following": res["rule_following_rate"],
                    "llm_agree_multi": res["llm_agreement_given_multi"],
                    "skill_share": tuple(res["skill_step_share"]),
                    "cond_marginal": tuple(res["condition_marginal"]),
                    "n_traj": res["num_trajectories"],
                }
            )
    if not rows:
        raise SystemExit(f"No train-env result JSONs for seed {seed} in {data_dir}")
    df = pd.DataFrame(rows)
    df["mode"] = pd.Categorical(df["mode"], MODE_ORDER, ordered=True)
    return df.sort_values(["game", "alg", "mode", "seed"])


def agg_scalar(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean and sample std across seeds."""
    out = (
        df.groupby(["game", "alg", "mode"], observed=True)[column]
        .agg(["mean", lambda s: s.std(ddof=1), "count"])
        .reset_index()
    )
    out.columns = ["game", "alg", "mode", "mean", "std", "n_seeds"]
    return out


def agg_vector(df: pd.DataFrame, column: str) -> dict:
    """(game, alg, mode) -> per-skill mean across seeds."""
    out = {}
    for key, grp in df.groupby(["game", "alg", "mode"], observed=True):
        out[key] = np.mean(np.stack(grp[column].to_numpy()), axis=0)
    return out


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #


def _grouped_bars(ax, stats, game, value_fmt="{:.0f}", log=False, ylabel=""):
    """Bars over the ablation ladder, one colour per algorithm."""
    sub = stats[stats["game"] == game]
    algs = [a for a in ALG_ORDER if a in set(sub["alg"])]
    modes = MODES_PRESENT  # shared across panels, so sharex stays truthful
    x = np.arange(len(modes))
    width = 0.78 / max(len(algs), 1)

    for i, alg in enumerate(algs):
        rows = sub[sub["alg"] == alg].set_index("mode")
        means = [rows["mean"].get(m, np.nan) for m in modes]
        errs = [rows["std"].get(m, np.nan) for m in modes]
        pos = x + (i - (len(algs) - 1) / 2) * width
        ax.bar(
            pos,
            means,
            width * 0.88,
            yerr=errs,
            capsize=2.5,
            color=ALG_COLOR[alg],
            edgecolor="white",
            linewidth=0.6,
            label=alg,
            error_kw={"elinewidth": 0.9, "ecolor": "0.3"},
        )
        # direct labels: identity is never carried by colour alone. Anchored
        # above the error bar so the two never collide.
        for p, m, e in zip(pos, means, errs):
            if np.isfinite(m):
                top = m + (e if np.isfinite(e) else 0.0)
                ax.annotate(
                    value_fmt.format(m),
                    (p, top),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=7,
                    color="0.25",
                )

    if log:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABEL[m] for m in modes], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(game)
    sns.despine(ax=ax)


def fig_performance(stats_ret, stats_len, prefix, figsize):
    fig, axes_ = plt.subplots(2, len(GAME_ORDER), figsize=figsize, sharex=True,
                              squeeze=False)
    for col, game in enumerate(GAME_ORDER):
        _grouped_bars(axes_[0, col], stats_ret, game, "{:.0f}", ylabel="Episode return")
        _grouped_bars(axes_[1, col], stats_len, game, "{:.0f}", ylabel="Episode length")
        axes_[1, col].set_title("")
    handles, labels = axes_[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, prefix, "performance")


def fig_skill_usage(shares, skills_by_key, prefix, figsize):
    """Stacked share of steps each option was active, per ablation."""
    keys = [(g, a) for g in GAME_ORDER for a in ALG_ORDER]
    fig, axes_ = plt.subplots(len(GAME_ORDER), len(ALG_ORDER), figsize=figsize,
                              squeeze=False)
    axes_ = np.atleast_2d(axes_)

    for idx, (game, alg) in enumerate(keys):
        ax = axes_[idx // len(ALG_ORDER), idx % len(ALG_ORDER)]
        modes = [m for m in MODES_PRESENT if (game, alg, m) in shares]
        if not modes:
            ax.axis("off")
            continue
        names = skills_by_key[(game, alg, modes[0])]
        y = np.arange(len(modes))[::-1]  # default at the top
        left = np.zeros(len(modes))
        for s, name in enumerate(names):
            vals = np.array([shares[(game, alg, m)][s] for m in modes])
            ax.barh(y, vals, left=left, height=0.62, color=SKILL_COLORS[s],
                    edgecolor="white", linewidth=1.2,
                    label=name.replace("_reward", "").replace("_", " "))
            for yy, v, l in zip(y, vals, left):
                if v > 0.06:  # labels stand in for the contrast WARN on light hues
                    ax.text(l + v / 2, yy, f"{100 * v:.0f}", ha="center",
                            va="center", fontsize=7, color="white")
            left = left + vals
        ax.set_yticks(y)
        ax.set_yticklabels([MODE_LABEL[m].replace("\n", " ") for m in modes])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of steps option was active")
        ax.set_title(f"{game} -- {alg}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
                  frameon=False, fontsize=7)
        sns.despine(ax=ax, left=True)
        ax.tick_params(axis="y", length=0)
    _save(fig, prefix, "skill_usage")


def fig_commitment(stats_dwell, stats_switch, prefix, figsize):
    fig, axes_ = plt.subplots(2, len(GAME_ORDER), figsize=figsize, sharex=True,
                              squeeze=False)
    for col, game in enumerate(GAME_ORDER):
        _grouped_bars(axes_[0, col], stats_dwell, game, "{:.0f}", log=True,
                      ylabel="Mean dwell (steps)")
        _grouped_bars(axes_[1, col], stats_switch, game, "{:.0f}", log=True,
                      ylabel="Switches per episode")
        axes_[1, col].set_title("")
    handles, labels = axes_[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, prefix, "commitment")


def fig_conditions(panels, prefix, figsize):
    titles = [
        ("multi_cond", "Steps with >1 condition active"),
        ("rule_following", "Executed option's condition held"),
        ("llm_agree_multi", "Agrees with LLM rule (contested)"),
    ]
    fig, axes_ = plt.subplots(len(GAME_ORDER), len(titles), figsize=figsize,
                              sharex=True, sharey=True, squeeze=False)
    axes_ = np.atleast_2d(axes_)
    for r, game in enumerate(GAME_ORDER):
        for c, (col, title) in enumerate(titles):
            ax = axes_[r, c]
            _grouped_bars(ax, panels[col], game, "{:.2f}",
                          ylabel=game if c == 0 else "")
            ax.set_ylim(0, 1.15)
            ax.set_title(title if r == 0 else "", fontsize=9)
    handles, labels = axes_[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, prefix, "conditions")


def _save(fig, prefix, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    for ext in ("pdf", "svg"):
        out = f"{prefix}_{name}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #


def write_table(df, shares, skills_by_key, prefix):
    rows = []
    for (game, alg, mode), grp in df.groupby(["game", "alg", "mode"], observed=True):
        names = skills_by_key[(game, alg, mode)]
        share = shares[(game, alg, mode)]
        row = {
            "game": game,
            "algorithm": alg,
            "ablation": mode,
            "seeds": len(grp),
            "return": f'{grp["episode_return"].mean():.0f} ± {grp["episode_return"].std(ddof=1):.0f}',
            "ep_length": f'{grp["episode_length"].mean():.0f}',
            "dwell": f'{grp["dwell"].mean():.1f}',
            "switches/ep": f'{grp["switches_per_ep"].mean():.0f}',
            ">1 cond": f'{grp["multi_cond"].mean():.3f}',
            "rule_following": f'{grp["rule_following"].mean():.3f}',
            "llm_agree_multi": f'{grp["llm_agree_multi"].mean():.3f}',
        }
        # positional columns: the two games have different option names, so
        # naming the columns after them leaves half the table empty
        row["options"] = " / ".join(n.replace("_reward", "") for n in names)
        for i, val in enumerate(share):
            row[f"share_{i + 1}"] = f"{val:.3f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(f"{prefix}_summary.csv", index=False)
    # hand-rolled markdown: pandas.to_markdown needs `tabulate`, which is not a
    # dependency of this project
    cols = list(table.columns)
    widths = [max(len(c), *(len(str(v)) for v in table[c])) for c in cols]
    def _row(vals):
        return "| " + " | ".join(str(v).ljust(w) for v, w in zip(vals, widths)) + " |"
    lines = [_row(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [_row(r) for r in table.itertuples(index=False)]
    with open(f"{prefix}_summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {prefix}_summary.csv and {prefix}_summary.md")
    return table


# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("outputs/activations"))
    p.add_argument("--seed", type=int, default=3,
                   help="which trained checkpoint (SEED) to plot")
    p.add_argument("--modes", default=None,
                   help="comma-separated subset of the ablation ladder to plot "
                        f"(default: all present; known: {','.join(MODE_ORDER)})")
    p.add_argument("--prefix", default=None,
                   help="output path prefix (default: next to this script)")
    p.add_argument("--figsize", type=parse_figsize, default="13x5")
    return p.parse_args()


def main():
    args = parse_args()
    prefix = args.prefix or str(Path(__file__).parent / "activations")

    df = load_records(args.data_dir, args.seed)
    if args.modes:
        wanted = [m.strip() for m in args.modes.split(",")]
        df = df[df["mode"].isin(wanted)]
        df["mode"] = df["mode"].cat.remove_unused_categories()
    print(f"Loaded {len(df)} (game, alg, ablation, seed) records from {args.data_dir}")
    print(df.groupby(["game", "alg", "mode"], observed=True).size().to_string())

    # narrow the game axis to what was actually loaded, so a single-game data
    # directory does not render an empty panel
    GAME_ORDER[:] = [g for g in GAME_ORDER if g in set(df["game"])]
    MODES_PRESENT[:] = [m for m in MODE_ORDER if m in set(df["mode"])]
    missing = [
        f"{g}/{a}/{m}"
        for g in df["game"].unique()
        for a in df.loc[df["game"] == g, "alg"].unique()
        for m in MODES_PRESENT
        if df[(df["game"] == g) & (df["alg"] == a) & (df["mode"] == m)].empty
    ]
    if missing:
        print(f"WARNING: {len(missing)} missing cell(s): {', '.join(missing)}")

    shares = agg_vector(df, "skill_share")
    skills_by_key = {k: g["skills"].iloc[0]
                     for k, g in df.groupby(["game", "alg", "mode"], observed=True)}

    with plt.rc_context(OLD_STYLE_RC):
        fig_performance(agg_scalar(df, "episode_return"),
                        agg_scalar(df, "episode_length"), prefix, args.figsize)
        fig_skill_usage(shares, skills_by_key, prefix,
                        (args.figsize[0], args.figsize[1] * 1.15))
        fig_commitment(agg_scalar(df, "dwell"), agg_scalar(df, "switches_per_ep"),
                       prefix, args.figsize)
        fig_conditions({c: agg_scalar(df, c)
                        for c in ("multi_cond", "rule_following", "llm_agree_multi")},
                       prefix, (args.figsize[0] * 1.2, args.figsize[1]))

    table = write_table(df, shares, skills_by_key, prefix)
    print()
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
