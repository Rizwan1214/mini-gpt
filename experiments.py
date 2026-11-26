"""Hyperparameter experiments: ek hi model, settings badal kar, nateeje compare karo.

Har experiment baseline se sirf EK cheez badalta hai, taake pata chale us cheez ka kya asar hai.

    python3 experiments.py                  (saare, har ek 1500 steps, ~6-8 minute)
    python3 experiments.py --steps 500      (jaldi, kam pakka)
    python3 experiments.py --only baseline no_attention

Nateeje: experiments/results.md (table), experiments/val_loss.svg (chart),
aur har run ka experiments/<naam>.csv.
"""

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from data import CharData, pick_device
from gpt import GPT, Config, train_model

OUT = Path(__file__).parent / "experiments"

# (naam, kya badla, config mein tabdeeli, learning rate)
EXPERIMENTS = [
    ("baseline", "4 layers, d=128, context 64, dropout 0.1, lr 1e-3", {}, 1e-3),
    ("no_dropout", "dropout 0.1 → 0", {"dropout": 0.0}, 1e-3),
    ("layers_2", "4 → 2 layers", {"n_layers": 2}, 1e-3),
    ("layers_6", "4 → 6 layers", {"n_layers": 6}, 1e-3),
    ("lr_low", "learning rate 1e-3 → 1e-4", {}, 1e-4),
    ("lr_high", "learning rate 1e-3 → 1e-2", {}, 1e-2),
    ("context_16", "context 64 → 16 characters", {"block_size": 16}, 1e-3),
    ("no_attention", "attention hata di (sirf MLP)", {"use_attention": False}, 1e-3),
]

COLORS = ["#14161c", "#d9480f", "#2b8a3e", "#1971c2", "#9c36b5", "#c2255c", "#0c8599", "#868e96"]


def write_svg(runs: list[dict], path: Path) -> None:
    """Val-loss curves as a dependency-free SVG line chart."""
    W, H, l, r, t, b = 760, 420, 52, 190, 20, 40
    max_step = max(s for run in runs for s, _, _ in run["log"])
    vals = [v for run in runs for _, _, v in run["log"]]
    lo, hi = 1.4, max(min(max(vals), 4.5), 2.0)
    x = lambda s: l + s / max_step * (W - l - r)
    y = lambda v: t + (hi - min(max(v, lo), hi)) / (hi - lo) * (H - t - b)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{l}" y="14" font-weight="bold">Validation loss (kam = behtar)</text>',
    ]
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        parts.append(f'<line x1="{l}" x2="{W - r}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e3e6ec"/>')
        parts.append(f'<text x="{l - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#5b6170">{v:.1f}</text>')
    for k in range(5):
        s = max_step * k / 4
        parts.append(f'<text x="{x(s):.1f}" y="{H - 18}" text-anchor="middle" fill="#5b6170">{int(s)}</text>')
    parts.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 2}" text-anchor="middle" fill="#5b6170">step</text>')
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        pts = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, _, v in run["log"])
        width = 3 if run["name"] == "baseline" else 1.8
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>')
        ly = t + 10 + i * 20
        parts.append(f'<line x1="{W - r + 14}" x2="{W - r + 34}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{W - r + 40}" y="{ly + 4}">{run["name"]} ({run["log"][-1][2]:.2f})</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--only", nargs="*", help="sirf ye experiments chalao")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    data = CharData()
    device = pick_device()
    base = Config(vocab_size=data.vocab_size)
    todo = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]
    print(f"{len(todo)} experiments × {args.steps} steps on {device}\n")

    runs = []
    for name, what, changes, lr in todo:
        cfg = replace(base, **changes)
        params = sum(p.numel() for p in GPT(cfg).parameters())
        print(f"▶ {name}: {what}  ({params / 1e6:.2f}M params)")
        _, log, secs = train_model(cfg, data, args.steps, lr=lr, eval_every=max(args.steps // 10, 1), device=device, verbose=False)
        step, train_loss, val_loss = log[-1]
        print(f"  train {train_loss:.3f}  val {val_loss:.3f}  gap {val_loss - train_loss:+.3f}  {secs:.0f}s\n")
        with open(OUT / f"{name}.csv", "w", newline="") as f:
            csv.writer(f).writerows([("step", "train", "val"), *log])
        runs.append({"name": name, "what": what, "params": params, "log": log, "secs": secs})

    base_val = next((r["log"][-1][2] for r in runs if r["name"] == "baseline"), None)
    lines = [
        f"| Experiment | Kya badla | Params | Train loss | Val loss | Val − Train | vs baseline | Waqt |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(runs, key=lambda r: r["log"][-1][2]):
        _, tr, va = r["log"][-1]
        delta = "–" if base_val is None or r["name"] == "baseline" else f"{va - base_val:+.3f}"
        lines.append(
            f"| `{r['name']}` | {r['what']} | {r['params'] / 1e6:.2f}M | {tr:.3f} | **{va:.3f}** | {va - tr:+.3f} | {delta} | {r['secs']:.0f}s |"
        )
    table = "\n".join(lines)
    (OUT / "results.md").write_text(f"# Experiments ({args.steps} steps each, {device})\n\n{table}\n")
    write_svg(runs, OUT / "val_loss.svg")
    print(table)
    print(f"\nSaved {OUT / 'results.md'} and {OUT / 'val_loss.svg'}")


if __name__ == "__main__":
    main()
