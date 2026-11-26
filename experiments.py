"""Hyperparameter experiments: ek hi model, settings badal kar, nateeje compare karo.

Har experiment baseline se sirf EK cheez badalta hai, taake pata chale us cheez ka kya asar hai.

    python3 experiments.py                  (saare, har ek 1500 steps, ~5 minute)
    python3 experiments.py --steps 500      (jaldi, kam pakka)
    python3 experiments.py --only baseline no_attention
    python3 experiments.py --seeds 3 --only baseline no_dropout --out-dir experiments/seeds

--seeds N: har experiment N alag seeds (alag shuruaati weights aur batches) se chalta hai.
Table mein val loss "mean ± std" aata hai, aur "Pakka?" column batata hai ke baseline se
farq shor (noise) se bada hai ya nahi.

Nateeje (--out-dir, default experiments/): results.md (table), val_loss.svg (chart),
aur har experiment ki <naam>.csv (seeds ka average).
"""

import argparse
import csv
import math
import statistics
from dataclasses import replace
from pathlib import Path

from data import TextData, pick_device
from gpt import GPT, Config, train_model

OUT = Path(__file__).parent / "experiments"
BASE_SEED = 1337

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
    # Neeche ki hadd 1.4, lekin data us se neeche jaye (jaise fine-tuning ~1.2) to aur neeche
    lo = min(1.4, math.floor((min(vals) - 0.05) * 10) / 10)
    hi = max(min(max(vals), 4.5), 2.0)
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
        parts.append(f'<text x="{l - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#5b6170">{v:.2f}</text>')
    for k in range(5):
        s = max_step * k / 4
        parts.append(f'<text x="{x(s):.1f}" y="{H - 18}" text-anchor="middle" fill="#5b6170">{int(s)}</text>')
    parts.append(f'<text x="{(l + W - r) / 2:.0f}" y="{H - 2}" text-anchor="middle" fill="#5b6170">step</text>')
    names = [e[0] for e in EXPERIMENTS]
    for i, run in enumerate(runs):
        # Har experiment ka rang hamesha ek hi rahe, chahe --only se kuch hi chalein
        color = COLORS[names.index(run["name"]) % len(COLORS)] if run["name"] in names else COLORS[i % len(COLORS)]
        pts = " ".join(f"{x(s):.1f},{y(v):.1f}" for s, _, v in run["log"])
        width = 3 if run["name"] == "baseline" else 1.8
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>')
        ly = t + 10 + i * 20
        parts.append(f'<line x1="{W - r + 14}" x2="{W - r + 34}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{W - r + 40}" y="{ly + 4}">{run["name"]} ({run["log"][-1][2]:.2f})</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def mean_log(logs: list[list[tuple]]) -> list[tuple]:
    """Kai seeds ke logs ka step-by-step average: [(step, train, val), ...]."""
    return [
        (rows[0][0], statistics.fmean(r[1] for r in rows), statistics.fmean(r[2] for r in rows))
        for rows in zip(*logs)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--only", nargs="*", help="sirf ye experiments chalao")
    ap.add_argument("--seeds", type=int, default=1, help="har experiment kitne alag seeds se chale")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="nateeje kahan save hon")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    data = TextData()
    device = pick_device()
    base = Config(vocab_size=data.vocab_size)
    seeds = [BASE_SEED + k for k in range(args.seeds)]
    todo = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]
    print(f"{len(todo)} experiments × {len(seeds)} seed(s) × {args.steps} steps on {device}\n")

    runs = []
    for name, what, changes, lr in todo:
        cfg = replace(base, **changes)
        params = sum(p.numel() for p in GPT(cfg).parameters())
        print(f"▶ {name}: {what}  ({params / 1e6:.2f}M params)")
        logs, secs = [], 0.0
        for seed in seeds:
            _, log, t = train_model(
                cfg, data, args.steps, lr=lr, eval_every=max(args.steps // 10, 1),
                seed=seed, device=device, verbose=False,
            )
            logs.append(log)
            secs += t
            _, tr, va = log[-1]
            print(f"  seed {seed}: train {tr:.3f}  val {va:.3f}  gap {va - tr:+.3f}  {t:.0f}s")
        finals = [log[-1][2] for log in logs]
        std = statistics.stdev(finals) if len(finals) > 1 else 0.0
        log = mean_log(logs)
        if len(seeds) > 1:
            print(f"  mean val {log[-1][2]:.3f} ± {std:.3f}")
        print()
        with open(out / f"{name}.csv", "w", newline="") as f:
            csv.writer(f).writerows([("step", "train", "val"), *log])
        runs.append({"name": name, "what": what, "params": params, "log": log, "std": std, "finals": finals, "secs": secs})

    multi = len(seeds) > 1
    baseline = next((r for r in runs if r["name"] == "baseline"), None)
    head = "| Experiment | Kya badla | Params | Train loss | Val loss | Val − Train | vs baseline |"
    rule = "|---|---|---|---|---|---|---|"
    if multi:
        head += " Pakka? |"
        rule += "---|"
    lines = [head + " Waqt |", rule + "---|"]
    for r in sorted(runs, key=lambda r: r["log"][-1][2]):
        _, tr, va = r["log"][-1]
        val = f"**{va:.3f}** ± {r['std']:.3f}" if multi else f"**{va:.3f}**"
        row = f"| `{r['name']}` | {r['what']} | {r['params'] / 1e6:.2f}M | {tr:.3f} | {val} | {va - tr:+.3f} |"
        if baseline is None or r is baseline:
            row += " – |" + (" – |" if multi else "")
        else:
            delta = va - baseline["log"][-1][2]
            row += f" {delta:+.3f} |"
            if multi:
                # Mote taur par: farq dono ke shor (std) ke milaap se 2 guna bada ho to pakka maano
                noise = (r["std"] ** 2 + baseline["std"] ** 2) ** 0.5
                row += " haan |" if abs(delta) > 2 * noise else " shayad shor |"
        lines.append(row + f" {r['secs']:.0f}s |")

    table = "\n".join(lines)
    seed_note = f", {len(seeds)} seeds ({seeds[0]}–{seeds[-1]}), mean ± std" if multi else f", seed {seeds[0]}"
    (out / "results.md").write_text(f"# Experiments ({args.steps} steps each, {device}{seed_note})\n\n{table}\n")
    write_svg(runs, out / "val_loss.svg")
    print(table)
    print(f"\nSaved {out / 'results.md'} and {out / 'val_loss.svg'}")


if __name__ == "__main__":
    main()
