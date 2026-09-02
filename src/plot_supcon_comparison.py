# Affine Equivariant Features, the main implementation of my master thesis.
# Copyright (C) 2026 Hendrik Sauer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Compare SupCon against ProxyAnchoredSupCon on the overall validation FPR95.

Point this at a directory of training runs and it plots one curve per (network,
loss) arm, keeping only the networks that were trained BOTH ways — an arm without
its counterpart is not a comparison, so it is dropped (and reported)::

    .venv13/bin/python src/plot_supcon_comparison.py /raid/data/hsa/logs/2026_08_20_pa

Where the numbers come from: ``train_func`` carries the whole per-epoch curve set
in every checkpoint under ``checkpoint["plots"]`` — ``x`` (epochs, the training
curve's axis), ``x_val`` (the validation axis, fractional epochs when
``validation.validate_every_n_batches`` is set), ``y_train`` and ``y_val``, the
latter keyed ``<loss>@<validation-split>`` (e.g.
``FPR95@all`` for the track configs, ``FPR95@overall`` for the synthetic ones).
So no re-evaluation is needed; this only reads checkpoints. ``latest.pth`` is
preferred over ``best.pth`` because ``best.pth`` is written at the best epoch and
its curves therefore stop there.

Which arm a run belongs to comes from its ``cfg.yaml`` (``training.loss``), not
from its directory name, and so does its network label (``model.name`` plus the
params that actually distinguish the variants: head, n_rotations, learned_mask,
and the run's ``scale``). Two runs that differ only in the contrastive loss
therefore land on the same network key automatically, whatever the launcher
called their folders. ``--label-from dir`` switches the key to the directory
basename for layouts where the config does not tell the arms apart.

Encoding: colour = network type, line style = loss (solid = ProxyAnchoredSupCon,
dashed = SupCon), a ringed dot marks each curve's best epoch. The palette is
the validated eight-slot categorical set, assigned in fixed order and never
cycled — past eight networks the script stops and asks you to narrow the
selection with ``--include``/``--exclude`` instead of inventing a ninth hue.

The default metric is ``ProxyAnchoredFPR95`` — FPR95 over pdf<->image pairs only,
i.e. the matching the descriptor is actually deployed for. It scores a different
pair population than plain ``FPR95`` and the two are NOT comparable, so
``--metric`` picks one and never mixes them on the axis; pass ``--metric FPR95``
for the all-pairs number instead. If no run reports the requested one, the script
falls back to the other for the whole corpus and says so.
"""

import argparse
import collections
import csv
import math
import os
import re
import sys

import matplotlib
import torch
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt, ticker  # noqa: E402  (after the Agg backend switch)


# The two contrastive training losses this script compares, in plot order, each with
# the line style that identifies it. Anything else in `training.loss` is ignored.
ARMS = [
    ("SupCon", "SupCon", (0, (5, 2))),
    ("ProxyAnchoredSupCon", "PA-SupCon", "solid"),
]

# Validated eight-slot categorical palette (light surface), assigned in fixed order.
# Order is the colour-blind-safety mechanism, not cosmetics — do not shuffle or extend.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# The other FPR95 to reach for when a corpus reports none of the requested one — runs
# from before ProxyAnchoredFPR95 existed only carry the plain metric, and vice versa.
METRIC_FALLBACK = {"ProxyAnchoredFPR95": "FPR95", "FPR95": "ProxyAnchoredFPR95"}

# Validation splits that mean "the whole set" rather than one band; the first one a run
# actually reports wins. Track configs call it `all`, the synthetic ones `overall`, and
# a single unlabelled validation dataset yields a bare metric name.
OVERALL_SPLITS = ["all", "overall", ""]


def find_runs(roots):
    """Yield every training-run directory under `roots`, deepest-first by path.

    A run directory is one holding both a `cfg.yaml` and a `checkpoints/` folder with
    at least one of latest/best — exactly what `prepare_training` writes.
    """
    seen = set()
    runs = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise SystemExit(f"!! not a directory: {root}")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            if "cfg.yaml" not in filenames:
                continue
            ckpt_dir = os.path.join(dirpath, "checkpoints")
            if not os.path.isdir(ckpt_dir):
                continue
            if dirpath not in seen:
                seen.add(dirpath)
                runs.append(dirpath)
    return sorted(runs)


def load_cfg(run_dir):
    with open(os.path.join(run_dir, "cfg.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def loss_names(entries):
    """Names out of a config `loss:` list, tolerating a bare string entry."""
    names = []
    for entry in entries or []:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and "name" in entry:
            names.append(entry["name"])
    return names


def arm_of(cfg):
    """Which contrastive arm a run belongs to, from its training losses.

    ProxyAnchoredSupCon wins if both are configured (that is a proxy-anchored run with
    a plain-SupCon auxiliary, not a SupCon baseline).
    """
    names = set(loss_names(cfg.get("training", {}).get("loss")))
    for key, _, _ in reversed(ARMS):
        if key in names:
            return key
    return None


def network_label(cfg):
    """A short name for the architecture, from the config alone.

    Mirrors the launcher's network vocabulary (`logpolar`, `logpolar_fft`,
    `efficient8`, `steerable`, ... plus `+mask`) so the labels match how the runs were
    submitted, but derives it from what was actually built rather than the folder name.
    """
    model = cfg.get("model") or {}
    name = model.get("name", "?")
    params = model.get("params") or {}

    if name == "HardNetLogPolar":
        base = "logpolar"
    elif name == "HardNet":
        base = "hardnet"
    elif name == "BlobDescriptorEfficient":
        base = f"efficient{params.get('n_rotations', '?')}"
    elif name == "BlobDescriptorNoStride":
        base = "steerable"
    else:
        base = name

    # Angular head: only the log-polar family has one, and `maxpool` is its default.
    head = params.get("head")
    if head and head not in ("maxpool", "attention"):
        base = f"{base}/{head}"
    if params.get("learned_mask"):
        base += "+mask"
    if params.get("cascade"):
        base += "/cascade" + ("-late" if params.get("cascade_late_weight") else "")

    scale = cfg.get("scale")
    if scale is not None:
        # `scale` may still be a raw string from an un-resolved interpolation.
        try:
            scale = f"{float(scale):g}"
        except (TypeError, ValueError):
            scale = str(scale)
        base += f" s{scale}"
    return base


def read_curve(run_dir, metric, split, checkpoint="auto"):
    """Return (epochs, values, metric_key, available_keys, checkpoint_path).

    `latest.pth` first: `best.pth` is saved at the best epoch, so its `plots` stop
    there and the curve would be silently truncated.
    """
    order = {"auto": ["latest", "best"], "latest": ["latest"], "best": ["best"]}[checkpoint]
    for stem in order:
        path = os.path.join(run_dir, "checkpoints", f"{stem}.pth")
        if not os.path.isfile(path):
            continue
        try:
            # mmap keeps the (large) state dicts off the heap; only `plots` is touched.
            ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, ValueError, TypeError):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        plots = ckpt.get("plots") if isinstance(ckpt, dict) else None
        if not plots:
            return None, None, None, [], path
        y_val = plots.get("y_val", {})
        keys = sorted(y_val)
        key = pick_metric_key(keys, metric, split)
        if key is None:
            return None, None, None, keys, path
        # `x_val` is the validation curve's own axis (it has more points than `x`
        # when sub-epoch validation is on); older checkpoints only carry `x`.
        x = list(plots.get("x_val") or plots.get("x", []))
        y = [float(v) for v in y_val[key]]
        n = min(len(x), len(y))
        return x[:n], y[:n], key, keys, path
    return None, None, None, [], None


def pick_metric_key(keys, metric, split):
    """Find `<metric>@<split>` among a run's reported validation curves."""
    by_split = {}
    for key in keys:
        name, _, label = key.partition("@")
        if name == metric:
            by_split[label] = key
    if not by_split:
        return None
    if split is not None:
        return by_split.get(split)
    for candidate in OVERALL_SPLITS:
        if candidate in by_split:
            return by_split[candidate]
    return None


def best_of(values):
    """(best value, its index) over the finite entries, or (nan, None)."""
    finite = [(v, i) for i, v in enumerate(values) if math.isfinite(v)]
    if not finite:
        return float("nan"), None
    return min(finite)


def matches(patterns, text):
    return any(re.search(p, text) for p in patterns)


def disambiguate(records, skipped):
    """Give colliding runs distinct network labels instead of dropping them.

    The label is derived from the config's model block, which does not name every
    hyperparameter a sweep might vary — a mask-resolution or a learning-rate sweep
    puts several genuinely different runs on one label. Where that happens the run's
    own directory name is appended, and to BOTH arms of that label, so a launcher
    that names its supcon/pa folders alike still pairs up.
    """
    grouped = collections.defaultdict(list)
    for record in records:
        grouped[(record["network"], record["arm"])].append(record)
    colliding = {label for (label, _), group in grouped.items() if len(group) > 1}
    if not colliding:
        return records

    for label in sorted(colliding):
        skipped.append((label, "several runs share this label — appending the run "
                               "directory name to tell them apart"))
    for record in records:
        if record["network"] in colliding:
            record["network"] = f"{record['network']} [{os.path.basename(record['run_dir'])}]"

    # Same label AND same directory name in one arm: nothing left to distinguish them
    # by, so keep the newer checkpoint and say which one lost.
    kept = {}
    for record in sorted(records, key=lambda r: r["mtime"]):
        key = (record["network"], record["arm"])
        if key in kept:
            skipped.append((kept[key]["rel"], f"duplicate of {record['rel']} for "
                                              f"{record['network']} / {record['arm']}"
                                              " — kept the newer checkpoint"))
        kept[key] = record
    return list(kept.values())


def collect(args):
    """Load every run under `args.run_dir` into per-(network, arm) records."""
    records = []
    skipped = []
    metric_keys_seen = set()

    for run_dir in find_runs(args.run_dir):
        rel = os.path.relpath(run_dir, os.path.abspath(os.path.commonpath(
            [os.path.abspath(d) for d in args.run_dir])))
        if args.include and not matches(args.include, rel):
            continue
        if args.exclude and matches(args.exclude, rel):
            continue

        try:
            cfg = load_cfg(run_dir)
        except yaml.YAMLError as exc:
            skipped.append((rel, f"unreadable cfg.yaml ({exc.__class__.__name__})"))
            continue

        arm = arm_of(cfg)
        if arm is None:
            skipped.append((rel, "no SupCon/ProxyAnchoredSupCon in training.loss"))
            continue

        label = network_label(cfg) if args.label_from == "cfg" else os.path.basename(run_dir)
        x, y, key, keys, ckpt_path = read_curve(run_dir, args.metric, args.split, args.checkpoint)
        metric_keys_seen.update(keys)
        if x is None:
            want = args.metric + (f"@{args.split}" if args.split is not None else "")
            skipped.append((rel, f"no {want} curve" + (f" (has: {', '.join(keys)})" if keys else "")))
            continue

        records.append({
            "network": label, "arm": arm, "run_dir": run_dir, "rel": rel,
            "x": x, "y": y, "metric_key": key, "checkpoint": ckpt_path,
            "mtime": os.path.getmtime(ckpt_path),
        })

    return disambiguate(records, skipped), skipped, sorted(metric_keys_seen)


def pair_up(records):
    """Keep only networks present in BOTH arms; return (paired, unpaired-notes)."""
    arms_by_network = collections.defaultdict(dict)
    for record in records:
        arms_by_network[record["network"]][record["arm"]] = record

    wanted = [key for key, _, _ in ARMS]
    paired, notes = {}, []
    for label in sorted(arms_by_network):
        present = arms_by_network[label]
        missing = [a for a in wanted if a not in present]
        if missing:
            notes.append(f"{label}: only {', '.join(sorted(present))} — missing {', '.join(missing)}")
            continue
        paired[label] = present
    return paired, notes


def plot(paired, args, metric_key):
    networks = sorted(paired)
    if len(networks) > len(SERIES_COLORS):
        raise SystemExit(
            f"!! {len(networks)} network types but the categorical palette has "
            f"{len(SERIES_COLORS)} slots (a 9th hue would not be colour-blind safe).\n"
            f"   Narrow the selection with --include/--exclude, e.g. --include 'logpolar'.\n"
            f"   Found: {', '.join(networks)}")

    colors = dict(zip(networks, SERIES_COLORS))
    styles = {key: style for key, _, style in ARMS}
    arm_names = {key: name for key, name, _ in ARMS}

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    max_epoch = 0
    for label in networks:
        for arm, _, _ in ARMS:
            record = paired[label][arm]
            x, y = record["x"], record["y"]
            max_epoch = max(max_epoch, max(x, default=0))
            ax.plot(x, y, color=colors[label], linestyle=styles[arm], linewidth=1.8,
                    solid_capstyle="round", zorder=3)
            best, index = best_of(y)
            if index is not None:
                # Ringed dot on the best epoch: the number the run is judged by, and
                # the ring keeps it readable where two curves cross.
                ax.plot([x[index]], [best], marker="o", markersize=6, color=colors[label],
                        markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="none",
                        zorder=4)

    if not args.linear:
        # Log scale (FPR95 spans decades), but labelled as plain numbers on a few
        # minor ticks as well: a decade axis showing only `10^-1` says nothing about
        # where in [0.02, 0.09] a run actually landed, which is the whole comparison.
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(2, 3, 5)))
        plain = ticker.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
    ax.set_xlim(0, max_epoch)
    ax.set_xlabel("Epoch", color=INK_SECONDARY)
    ax.set_ylabel(f"Validation {metric_key}", color=INK_SECONDARY)
    ax.set_title(args.title or f"SupCon vs. proxy-anchored SupCon — {metric_key}",
                 color=INK_PRIMARY, fontsize=11, loc="left", pad=12)

    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.grid(True, which="minor", color=GRIDLINE, linewidth=0.3, alpha=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)

    # Two legends, because the chart uses two channels: colour says which network,
    # dash pattern says which loss. Neither is readable from the other's key.
    color_handles = [plt.Line2D([], [], color=colors[n], linewidth=2.4, label=n) for n in networks]
    style_handles = [plt.Line2D([], [], color=INK_SECONDARY, linewidth=1.8, linestyle=s,
                                label=arm_names[k]) for k, _, s in ARMS]
    first = ax.legend(handles=color_handles, title="Network", loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8,
                      title_fontsize=8, labelcolor=INK_SECONDARY)
    first.get_title().set_color(INK_SECONDARY)
    second = ax.legend(handles=style_handles, title="Loss", loc="lower left",
                       bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=8,
                       title_fontsize=8, labelcolor=INK_SECONDARY)
    second.get_title().set_color(INK_SECONDARY)
    ax.add_artist(first)

    # Both legends sit outside the axes, so they have to be named explicitly for the
    # tight bounding box — otherwise the longer network labels are cropped off.
    fig.savefig(args.out, facecolor=SURFACE, bbox_inches="tight",
                bbox_extra_artists=(first, second))
    plt.close(fig)


def summarize(paired, metric_key):
    """Table of best-per-arm plus the PA-SupCon delta; also the chart's table view.

    Three of the palette's light-mode slots sit below 3:1 contrast, so the numbers
    have to be legible somewhere other than the colours — this is that somewhere.
    """
    rows = []
    for label in sorted(paired):
        row = {"network": label, "metric": metric_key}
        for arm, name, _ in ARMS:
            record = paired[label][arm]
            best, index = best_of(record["y"])
            row[f"{name}_best"] = best
            row[f"{name}_epoch"] = record["x"][index] if index is not None else None
            row[f"{name}_run"] = record["rel"]
        base, proxy = row["SupCon_best"], row["PA-SupCon_best"]
        row["delta"] = proxy - base
        row["rel_change"] = (proxy - base) / base if base else float("nan")
        rows.append(row)

    width = max((len(r["network"]) for r in rows), default=7)
    print(f"\nBest {metric_key} per network (lower is better)\n")
    print(f"  {'network'.ljust(width)}  {'SupCon':>10}  {'PA-SupCon':>10}  {'delta':>10}  {'change':>8}")
    print(f"  {'-' * width}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for row in rows:
        print(f"  {row['network'].ljust(width)}  {row['SupCon_best']:>10.4g}  "
              f"{row['PA-SupCon_best']:>10.4g}  {row['delta']:>+10.4g}  "
              f"{row['rel_change'] * 100:>+7.1f}%")
    wins = sum(1 for r in rows if r["delta"] < 0)
    print(f"\n  PA-SupCon wins on {wins}/{len(rows)} networks.")
    return rows


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        epilog="Runs are paired by network, not by folder name — see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="+",
                        help="directory (or directories) of training runs, searched recursively")
    parser.add_argument("-o", "--out", default="fpr95_supcon_vs_pa.svg",
                        help="output figure path; the extension picks the format (default: %(default)s)")
    parser.add_argument("--metric", default="ProxyAnchoredFPR95",
                        help="validation metric to plot (default: %(default)s, the "
                             "pdf<->image pairs the descriptor is deployed for). It "
                             "scores a different pair population than plain FPR95 and "
                             "the two are NOT comparable")
    parser.add_argument("--split", default=None,
                        help="validation split to read, i.e. the part after '@' "
                             f"(default: the first of {'/'.join(s or '<unlabelled>' for s in OVERALL_SPLITS)})")
    parser.add_argument("--checkpoint", choices=["auto", "latest", "best"], default="auto",
                        help="which checkpoint to read the curves from; 'best' truncates them "
                             "at its own epoch (default: %(default)s = latest, else best)")
    parser.add_argument("--label-from", choices=["cfg", "dir"], default="cfg",
                        help="derive the network label from the config's model block (default) "
                             "or from the run's directory name")
    parser.add_argument("--include", action="append", default=[], metavar="REGEX",
                        help="only runs whose path matches (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], metavar="REGEX",
                        help="drop runs whose path matches (repeatable)")
    parser.add_argument("--linear", action="store_true",
                        help="linear y axis instead of the default log scale")
    parser.add_argument("--csv", default=None, metavar="PATH",
                        help="also write the summary table as CSV")
    parser.add_argument("--title", default=None, help="override the figure title")
    parser.add_argument("--width", type=float, default=8.0, help="figure width in inches")
    parser.add_argument("--height", type=float, default=4.5, help="figure height in inches")
    parser.add_argument("--dpi", type=int, default=150, help="figure dpi for raster formats")
    args = parser.parse_args()

    records, skipped, keys_seen = collect(args)

    # A whole-corpus fallback, not a per-run one: mixing FPR95 and ProxyAnchoredFPR95
    # on one axis would compare different pair populations. Older runs predate the
    # proxy-anchored metric and report only plain FPR95, hence the swap.
    fallback = METRIC_FALLBACK.get(args.metric)
    if not records and fallback and any(k.partition("@")[0] == fallback for k in keys_seen):
        print(f"!! no {args.metric} curves found, but {fallback} is reported — retrying with it.\n"
              "   The two score different pair populations and are not comparable.",
              file=sys.stderr)
        args.metric = fallback
        records, skipped, keys_seen = collect(args)

    if skipped:
        print("Skipped:", file=sys.stderr)
        for rel, why in skipped:
            print(f"  {rel}: {why}", file=sys.stderr)

    paired, notes = pair_up(records)
    if notes:
        print("\nNot plotted (no SupCon/PA-SupCon pair):", file=sys.stderr)
        for note in notes:
            print(f"  {note}", file=sys.stderr)

    if not paired:
        raise SystemExit("\n!! no network has both a SupCon and a ProxyAnchoredSupCon run "
                         "— nothing to compare.")

    metric_key = next(iter(next(iter(paired.values())).values()))["metric_key"]
    # Table before figure: if there are too many networks for the palette, the numbers
    # are still worth having while you narrow the selection down.
    rows = summarize(paired, metric_key)
    plot(paired, args, metric_key)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"  table  -> {args.csv}")
    print(f"  figure -> {args.out}")


if __name__ == "__main__":
    main()
