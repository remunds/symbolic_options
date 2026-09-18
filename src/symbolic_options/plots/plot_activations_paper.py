"""Two condensed figures summarising the meta-policy ablation study.

The full `plot_activations.py` grid (7 rungs x 2 algorithms x 2 games) is too
dense to publish. This script draws the two figures that carry the argument:

  <prefix>_ladder.pdf/svg
      Return relative to the unablated policy, across the commitment-controlled
      ablation ladder. Normalising by each agent's own default puts Seaquest and
      Kangaroo -- whose raw returns differ by ~2x -- on one axis, so the four
      agents can be compared directly. Colour = game, marker/line = algorithm.

  <prefix>_decisions.pdf/svg
      What the meta-policy does in each symbolic situation: one row per subset
      of skill conditions that holds, showing how often that situation occurs
      and which skill is executed in it. This is the mechanism behind the
      ladder: it shows *where* the meta-policy's choices depart from the rules.

Records are keyed by (game, algorithm, ablation, seed); when several --data-dir
are given, later ones win. That is how the Kangaroo EPS_TEST=0.1 re-run (which
broke the environment's determinism) replaces the original Kangaroo numbers
while leaving Seaquest, which never had that problem, untouched.

Usage:
    python src/symbolic_options/plots/plot_activations_paper.py \\
        --data-dir outputs/activations --data-dir outputs/activations_eps01
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from plots_old_style import OLD_STYLE_RC, get_algorithm_from_name, parse_figsize  # noqa: E402

# the commitment-controlled ladder: every random rung holds its draw for 20
# steps, so a drop in return is a worse *choice*, not lost temporal abstraction
LADDER = ["default", "tiebreak+hold", "masked+hold", "uniform+hold"]
LADDER_LABEL = {
    "default": "default\n(meta-policy)",
    "tiebreak+hold": "random in\nambiguous states",
    "masked+hold": "random among\nenabled skills",
    "uniform+hold": "random among\nall skills",
}
GAMES = ["Seaquest", "Kangaroo"]
ALGS = ["NEXUS (symbolic)", "NEXUS (nesy)"]

_CB = sns.color_palette("colorblind").as_hex()
GAME_COLOR = {"Seaquest": _CB[0], "Kangaroo": _CB[3]}      # blue, vermillion
ALG_MARKER = {ALGS[0]: "o", ALGS[1]: "s"}                   # composite encoding:
ALG_LINE = {ALGS[0]: "-", ALGS[1]: "--"}                    # identity never colour-only
SKILL_COLORS = [_CB[0], _CB[1], _CB[2], _CB[4], _CB[9]]

# the focused figure: the hybrid meta-policy, and the one ablation that isolates
# its arbitration (random among enabled skills, held 20 steps, only where the
# symbolic rules leave more than one skill enabled)
NESY = ALGS[1]
AMBIG = "tiebreak+hold"
COND_COLOR = _CB[1]   # "condition holds" -- the opportunity
USED_COLOR = _CB[0]   # "skill executed" -- what the meta-policy did with it

# the repo's "aligned metric" (cf. ALIGNED_LABELS in plots_old_style): the
# goal-directed outcome behind the raw game reward, on a twin axis as in
# dual_metric_plot. Purple, because blue/orange/green already mean skills here.
# These are environment metrics read at episode end (env_info_*), not reward
# functions -- the same quantities new_plots.py pulls from wandb as
# test_clean/total_divers_collected and test_clean/game_progress, so these
# numbers are directly comparable with the repo's other figures.
# Drawn in the same orange as the rest of the figure, for visual consistency.
ALIGNED = {
    "Seaquest": ("total_divers_collected", "Divers Collected"),
    "Kangaroo": ("game_progress", "Game Progress"),
}
ALIGNED_COLOR = COND_COLOR

plt.rcParams.update({"figure.dpi": 150})


def load(data_dirs, seeds, eval_env="train"):
    """(game, alg, mode) -> list of result blobs, one per SEED.

    Several --data-dir are merged with later ones overriding earlier ones for
    the same (game, alg, mode, seed). Several --seeds are *pooled*: a run's
    vmaps are 3 network initialisations within one SEED, so seeds 0 and 3
    together give the 6 seeds the repo's other figures aggregate over.
    """
    by_seed = {}
    for data_dir in data_dirs:
        for path in sorted(Path(data_dir).glob("*.json")):
            blob = json.loads(Path(path).read_text())
            if blob.get("eval_env", "").split("_")[0] != eval_env:
                continue
            if int(blob.get("seed", -1)) not in seeds:
                continue
            mode = blob.get("random_meta") or "default"
            if int(blob.get("random_hold", 1) or 1) > 1:
                mode += "+hold"
            key = (blob["env_name"], get_algorithm_from_name(blob["alg_name"]), mode)
            by_seed[(key, int(blob["seed"]))] = blob
    out = {}
    for (key, seed), blob in sorted(by_seed.items(), key=lambda kv: kv[0][1]):
        out.setdefault(key, []).append(blob)
    return out


def _seed_values(blobs, key):
    """One value per vmap, pooled over every SEED present."""
    if isinstance(blobs, dict):          # a single blob
        blobs = [blobs]
    return np.array([r[key] for b in blobs for r in b["per_vmap"].values()],
                    dtype=float)


def _vec_mean(blobs, key):
    """Mean of a per-skill (or per-pattern) vector over all pooled vmaps."""
    if isinstance(blobs, dict):
        blobs = [blobs]
    stack = np.array([r[key] for b in blobs for r in b["per_vmap"].values()],
                     dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


# --------------------------------------------------------------------------- #


def fig_ladder(data, prefix, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(LADDER))

    for game in GAMES:
        for alg in ALGS:
            base = data.get((game, alg, "default"))
            if not base:
                continue
            ref = _seed_values(base, "episode_return_mean").mean()
            ys, es = [], []
            for mode in LADDER:
                blob = data.get((game, alg, mode))
                if not blob:
                    ys.append(np.nan)
                    es.append(np.nan)
                    continue
                vals = _seed_values(blob, "episode_return_mean") / ref
                ys.append(vals.mean())
                es.append(vals.std(ddof=1))
            ax.errorbar(
                x, ys, yerr=es,
                color=GAME_COLOR[game], marker=ALG_MARKER[alg], linestyle=ALG_LINE[alg],
                linewidth=2, markersize=7, capsize=3, elinewidth=0.9,
                label=f"{game} - {alg.replace('NEXUS ', '').strip('()')}",
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            )
            # direct end-label so the four series are identifiable without the key
            if np.isfinite(ys[-1]):
                ax.annotate(f" {ys[-1]:.2f}", (x[-1], ys[-1]), fontsize=7,
                            color="0.25", va="center")

    ax.axhline(1.0, color="0.65", linewidth=1, linestyle=":", zorder=1)
    ax.annotate("unablated policy", (0.55, 1.0), xycoords=("axes fraction", "data"),
                fontsize=7, color="0.45", va="bottom", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([LADDER_LABEL[m] for m in LADDER], fontsize=8)
    ax.set_ylabel("Episode return\n(relative to unablated)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    sns.despine(ax=ax)
    _save(fig, prefix, "ladder")


def fig_decisions(data, prefix, figsize):
    """One row per condition subset: how often it occurs, what is executed."""
    cells = [
        (g, a)
        for g in GAMES
        for a in ALGS
        if data.get((g, a, "default"))
        and "pattern_share" in data[(g, a, "default")][0]["aggregate"]
    ]
    fig, axes_ = plt.subplots(len(GAMES), len(ALGS), figsize=figsize, squeeze=False)

    for idx, (game, alg) in enumerate(cells):
        ax = axes_[idx // len(ALGS), idx % len(ALGS)]
        blobs = data[(game, alg, "default")]
        names = [n.replace("_reward", "").replace("_", " ")
                 for n in blobs[0]["skill_names"]]
        share = _vec_mean(blobs, "pattern_share")
        choice = _vec_mean(blobs, "pattern_choice")

        keep = [p for p in np.argsort(-share) if share[p] > 5e-4]
        y = np.arange(len(keep))[::-1]

        left = np.zeros(len(keep))
        for s, name in enumerate(names):
            vals = np.nan_to_num(choice[keep, s])
            ax.barh(y, vals, left=left, height=0.6, color=SKILL_COLORS[s],
                    edgecolor="white", linewidth=1.2, label=name)
            for yy, v, l in zip(y, vals, left):
                if v > 0.08:
                    ax.text(l + v / 2, yy, f"{100 * v:.0f}", ha="center", va="center",
                            fontsize=7, color="white")
            left = left + vals

        labels = []
        for p in keep:
            on = [names[k] for k in range(len(names)) if p >> k & 1]
            labels.append(f"{' + '.join(on)}\n({100 * share[p]:.0f}% of steps)")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Skill executed in that situation", fontsize=8)
        ax.set_title(f"{game} — {alg}", fontsize=9)
        # the two panels in a row share a skill set, so one legend per row
        if idx % len(ALGS) == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(1.06, -0.22), ncol=3,
                      frameon=False, fontsize=7)
        sns.despine(ax=ax, left=True)
        ax.tick_params(axis="y", length=0)

    for j in range(len(cells), axes_.size):
        axes_[j // len(ALGS), j % len(ALGS)].axis("off")
    _save(fig, prefix, "decisions")


def fig_nesy(data, prefix, figsize):
    """The hybrid meta-policy in one figure: one row per game.

    Read left to right: what randomising *only* the ambiguous states costs;
    then how often each skill is permitted versus how often it is actually
    run; then, per subset of conditions, which skill is executed there.

    Styling follows the repo's other figures (tueplots neurips2024 + axes.lines
    via OLD_STYLE_RC, seaborn "white", bold axis labels, colorblind palette).
    """
    games = [g for g in GAMES if data.get((g, NESY, "default"))]
    # one subfigure per game: each carries a title spanning its whole row, which
    # states the game far more clearly than a title on one panel could
    fig = plt.figure(figsize=figsize, layout="constrained")
    subfigs = fig.subfigures(len(games), 1, hspace=0.06)
    subfigs = np.atleast_1d(subfigs)
    axes_ = {}
    titles = []
    for r, game in enumerate(games):
        sf = subfigs[r]
        # black, not the game colour: a coloured heading reads as a data
        # encoding here, since the bars already use colour to mean something.
        # No explicit y, so constrained_layout reserves room for the title.
        titles.append(sf.suptitle(game, fontweight="bold", fontsize=16,
                                  color="black"))
        # constrained_layout packs the columns against their decorations, where
        # tight_layout leaves each axes floating in an over-wide gridspec cell
        row = sf.subplots(1, 3, gridspec_kw={"width_ratios": [0.62, 1.05, 1.33]})
        for c in range(3):
            axes_[(r, c)] = row[c]

    for r, game in enumerate(games):
        blobs = data[(game, NESY, "default")]
        names = [n.replace("_reward", "").replace("_", " ")
                 for n in blobs[0]["skill_names"]]
        na = len(names)

        # ---- left: what randomising the ambiguous states costs ------------ #
        ax = axes_[r, 0]
        aligned_key, aligned_label = ALIGNED[game]
        ret, ret_e, ali, ali_e = [], [], [], []
        for mode in ("default", AMBIG):
            blobs_m = data[(game, NESY, mode)]
            vals = _seed_values(blobs_m, "episode_return_mean")
            ret.append(vals.mean())
            ret_e.append(vals.std(ddof=1))
            idx = blobs_m[0]["env_info_names"].index(aligned_key)
            a = np.array([r_["env_info_mean"][idx]
                          for b_ in blobs_m for r_ in b_["per_vmap"].values()],
                         dtype=float)
            ali.append(a.mean())
            ali_e.append(a.std(ddof=1))

        x = np.array([0, 1])
        ax2 = ax.twinx()  # dual axis, matching the repo's dual_metric_plot
        ax.bar(x - 0.2, ret, yerr=ret_e, width=0.4, capsize=3,
               color=USED_COLOR, edgecolor="white")
        ax2.bar(x + 0.2, ali, yerr=ali_e, width=0.4, capsize=3,
                color=ALIGNED_COLOR, edgecolor="white")
        # every bar is labelled: which axis a bar belongs to is otherwise
        # carried by colour alone, and the purple is below 3:1 on contrast
        for xpos, v, e in zip(x - 0.2, ret, ret_e):
            ax.annotate(f"{v:.0f}", (xpos, v + e), xytext=(0, 3), fontsize=7,
                        textcoords="offset points", ha="center", color="0.25")
        for xpos, v, e in zip(x + 0.2, ali, ali_e):
            ax2.annotate(f"{v:.1f}", (xpos, v + e), xytext=(0, 3), fontsize=7,
                         textcoords="offset points", ha="center", color="0.25")

        ax.set_xticks(x)
        ax.set_xticklabels(["meta-\npolicy", "random in\nambiguous"])
        ax.set_ylabel("Episode Return", fontweight="bold", color=USED_COLOR)
        ax.tick_params(axis="y", labelcolor=USED_COLOR)
        ax2.set_ylabel(aligned_label, fontweight="bold", color=ALIGNED_COLOR)
        ax2.tick_params(axis="y", labelcolor=ALIGNED_COLOR)
        ax.set_ylim(0, max(v + e for v, e in zip(ret, ret_e)) * 1.32)
        ax2.set_ylim(0, max(v + e for v, e in zip(ali, ali_e)) * 1.32)
        ax.annotate(f"-{100 * (1 - ret[1] / ret[0]):.0f}%", (0.28, 0.93),
                    xycoords="axes fraction", ha="center", fontsize=10,
                    color=USED_COLOR)
        ax.annotate(f"-{100 * (1 - ali[1] / ali[0]):.0f}%", (0.78, 0.93),
                    xycoords="axes fraction", ha="center", fontsize=10,
                    color=ALIGNED_COLOR)

        # ---- middle: permitted vs actually executed ----------------------- #
        ax = axes_[r, 1]
        cond = _vec_mean(blobs, "condition_marginal")
        used = _vec_mean(blobs, "skill_step_share")
        y = np.arange(na)[::-1]
        h = 0.36
        for series, vals, color, lbl in (
            (0, cond, COND_COLOR, "condition holds"),
            (1, used, USED_COLOR, "skill executed"),
        ):
            ax.barh(y + (0.5 - series) * h, vals, height=h * 0.92, color=color,
                    edgecolor="white", label=lbl)
            for yy, v in zip(y + (0.5 - series) * h, vals):
                ax.annotate(f"{v:.2f}", (v, yy), xytext=(3, 0), fontsize=8,
                            textcoords="offset points", va="center", color="0.25")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("Share of Steps", fontweight="bold")
        ax.legend(frameon=False, loc="lower right")  # on every row
        ax.tick_params(axis="y", length=0)

        # ---- right: what it executes in each symbolic situation ----------- #
        ax = axes_[r, 2]
        share = _vec_mean(blobs, "pattern_share")
        choice = _vec_mean(blobs, "pattern_choice")
        keep = [p for p in np.argsort(-share) if share[p] > 5e-4]
        yy = np.arange(len(keep))[::-1]
        left = np.zeros(len(keep))
        for s_i, name in enumerate(names):
            vals = np.nan_to_num(choice[keep, s_i])
            ax.barh(yy, vals, left=left, height=0.6, color=SKILL_COLORS[s_i],
                    edgecolor="white", linewidth=1.2, label=name)
            for ypos, v, l in zip(yy, vals, left):
                if v > 0.08:
                    ax.text(l + v / 2, ypos, f"{100 * v:.0f}", ha="center",
                            va="center", fontsize=8, color="white")
            left = left + vals
        ax.set_yticks(yy)
        ax.set_yticklabels(
            [f"{' + '.join(names[k] for k in range(na) if p >> k & 1)}\n"
             f"({100 * share[p]:.0f}% of steps)" for p in keep], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("Skill Executed at Conditions", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
                  frameon=False)
        ax.tick_params(axis="y", length=0)

    fig.get_layout_engine().set(w_pad=0.02, wspace=0.02)

    # Draw once so constrained_layout settles, then freeze it: the divider is
    # positioned from the resulting geometry, so the layout must not move after.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    for r in range(1, len(games)):
        prev_bottom = min(
            to_fig.transform(axes_[(r - 1, c)].get_tightbbox(renderer))[0][1]
            for c in range(3)
        )
        title_top = to_fig.transform(
            titles[r].get_window_extent(renderer))[1][1]
        y = (prev_bottom + title_top) / 2
        fig.add_artist(Line2D([0.01, 0.99], [y, y], transform=fig.transFigure,
                              color="0.8", linewidth=1.0))

    _save(fig, prefix, "nesy", skip_layout=True)


def _save(fig, prefix, name, w_pad=None, skip_layout=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if not skip_layout:  # figures using constrained_layout lay themselves out
            fig.tight_layout(**({} if w_pad is None else {"w_pad": w_pad}))
    for ext in ("pdf", "svg"):
        fig.savefig(f"{prefix}_{name}.{ext}", bbox_inches="tight")
        print(f"Saved {prefix}_{name}.{ext}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", action="append", default=None,
                   help="repeatable; later directories override earlier ones")
    p.add_argument("--seeds", default="3",
                   help="comma-separated SEEDs to pool (e.g. 0,3)")
    p.add_argument("--eval-env", default="train",
                   help="which eval env's results to load (train / clean / ...)")
    p.add_argument("--prefix", default=None)
    p.add_argument("--figsize", type=parse_figsize, default="6x3.4")
    p.add_argument("--decisions-figsize", type=parse_figsize, default="9x6")
    p.add_argument("--nesy-figsize", type=parse_figsize, default="11x5")
    p.add_argument("--all", action="store_true",
                   help="also draw the ladder and full decision-map figures")
    args = p.parse_args()

    dirs = args.data_dir or ["outputs/activations"]
    prefix = args.prefix or str(Path(__file__).parent / "activations_paper")
    seeds = {int(x) for x in str(args.seeds).split(",") if x.strip() != ""}
    data = load(dirs, seeds, eval_env=args.eval_env)
    print(f"Loaded {len(data)} (game, alg, ablation) cells from {dirs}; "
          f"seeds {sorted(seeds)}, "
          f"{sum(len(v) for v in data.values())} runs pooled")

    missing = [
        k for k in data
        if k[2] == "default" and "pattern_share" not in data[k][0]["aggregate"]
    ]
    if missing:
        # only the decisions figure needs them; the ladder still needs these
        # cells as its normalisation reference, so they are not dropped
        print(f"NOTE: no pattern breakdown for {missing} -- re-run those cells "
              "with the current eval_activations.py to include them")

    with plt.rc_context(OLD_STYLE_RC):
        sns.set_style("white")  # as the repo's other figures do
        fig_nesy(data, prefix, args.nesy_figsize)
        if args.all:
            fig_ladder(data, prefix, args.figsize)
            fig_decisions(data, prefix, args.decisions_figsize)


if __name__ == "__main__":
    main()
