"""Extract console metrics and plot training curves without W&B access."""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/training_plots"))
    args = parser.parse_args()
    rows = {}
    for line in args.log.read_text(errors="replace").splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        match = re.search(r"\bstep:(\d+)\s+-\s+", line)
        if not match:
            continue
        values = re.findall(r"([^\s]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", line[match.end():])
        rows.setdefault(int(match[1]), {}).update({k: float(v) for k, v in values})
    if not rows:
        raise SystemExit("No step metrics found")
    args.output.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows.values() for k in row})
    with (args.output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step"] + keys)
        writer.writeheader()
        for step, values in sorted(rows.items()):
            writer.writerow({"step": step, **values})

    def series(ax, key, label=None):
        points = [(s, v[key]) for s, v in sorted(rows.items()) if key in v]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, label=label or key, linewidth=1.5,
                    marker="o" if key.startswith("val/") else None, markersize=3)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    panels = [
        ("Success rate", [("episode/success_rate", "Train"), ("val/success_rate", "Validation")]),
        ("Reward / validation score (different definitions)", [("episode/reward/mean", "Train episode reward"), ("val/text/test_score", "Validation test score")]),
        ("Actor losses", [("actor/pg_loss", "Policy gradient"), ("actor/kl_loss", "KL")]),
        ("Entropy", [("actor/entropy_loss", "Actor entropy")]),
        ("Mean episode length", [("episode/length/mean", "Episode length")]),
        ("Step time (includes validation when run)", [("timing_s/step", "Seconds")]),
    ]
    for ax, (title, metrics) in zip(axes.flat, panels):
        for key, label in metrics:
            series(ax, key, label)
        ax.set(title=title, xlabel="Training step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0, 0].set_ylim(0, 1.05)
    fig.suptitle(f"{args.log.name} | raw console metrics (rounded)", fontsize=16)
    for suffix in ["png", "svg"]:
        fig.savefig(args.output / f"overview.{suffix}", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for key in keys:
        if key.startswith("val/") and key.endswith("_success_rate"):
            series(ax, key, key.removeprefix("val/").removesuffix("_success_rate"))
    ax.set(title="Validation success rate by task", xlabel="Training step", ylabel="Success rate", ylim=(0, 1.05))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(args.output / "validation_tasks.png", dpi=160)
    plt.close(fig)
    val = [(s, v["val/success_rate"]) for s, v in sorted(rows.items()) if "val/success_rate" in v]
    summary = {"source": str(args.log), "steps": len(rows), "validation_points": len(val),
               "first_validation": val[0] if val else None,
               "last_validation": val[-1] if val else None,
               "best_validation": max(val, key=lambda p: p[1]) if val else None,
               "note": "Parsed rounded console values; no smoothing or interpolation of missing metrics."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
