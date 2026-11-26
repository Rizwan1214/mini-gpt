"""Tokenizers ka muqabla: same model, same steps, sirf tokenizer alag.

    python3 compare_tokenizers.py                     (char, bpe256, bpe512, bpe1024, gpt2; ~6 minute)
    python3 compare_tokenizers.py --only char bpe512

Zaroori baat: alag tokenizers ka loss seedha compare NAHI hota. BPE ka ek token kai characters
ka hota hai, is liye uska "per token" loss zyada hoga. Insaaf ke liye loss ko per character mein
badalte hain:  loss_per_char = loss_per_token / (characters per token).

Nateeje: experiments/tokenizers/results.md
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import torch

from data import TextData, pick_device
from gpt import GPT, Config, train_model
from tokenization import show_split

OUT = Path(__file__).parent / "experiments" / "tokenizers"
SENTENCE = "ROMEO: But soft, what light through yonder window breaks?"
TOKENIZERS = ["char", "bpe256", "bpe512", "bpe1024", "gpt2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--only", nargs="*", help="sirf ye tokenizers")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    rows, splits, samples = [], [], []

    for name in [t for t in TOKENIZERS if not args.only or t in args.only]:
        t0 = time.time()
        data = TextData(name)
        tok_secs = time.time() - t0
        cfg = replace(Config(), vocab_size=data.vocab_size)
        params = sum(p.numel() for p in GPT(cfg).parameters())
        emb_share = 2 * data.vocab_size * cfg.d_model / params  # token embedding + output layer
        cpt = data.val_chars_per_token
        print(f"▶ {name}: vocab {data.vocab_size}, {cpt:.2f} chars/token, {params / 1e6:.2f}M params "
              f"({emb_share:.0%} embeddings), tokenizer {tok_secs:.1f}s")

        model, log, secs = train_model(cfg, data, args.steps, eval_every=max(args.steps // 5, 1),
                                       device=device, verbose=False)
        _, train_tok, val_tok = log[-1]
        print(f"  val {val_tok:.3f}/token = {val_tok / cpt:.3f}/char  ({secs:.0f}s)\n")
        rows.append((name, data.vocab_size, params, emb_share, cpt, cfg.block_size * cpt, val_tok, val_tok / cpt, secs))
        splits.append((name, len(data.encode(SENTENCE)), show_split(data.tokenizer, SENTENCE)))

        # Har model se ~250 characters likhwao (same seed, same temperature)
        torch.manual_seed(0)
        model.eval()
        start = torch.tensor([data.encode("\n")[:1]], device=device)
        n_tokens = max(1, round(250 / cpt))
        samples.append((name, data.decode(model.generate(start, n_tokens, temperature=0.8)[0].tolist()).strip()))

    lines = [
        f"# Tokenizers ({args.steps} steps each, {device}, model: 4 layers, d=128, context 64 tokens)",
        "",
        "| Tokenizer | Vocab | Params | Embeddings ka hissa | Chars/token | Context (chars) | Val loss / token | **Val loss / char** | Waqt |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, vocab, params, share, cpt, ctx, vt, vc, secs in sorted(rows, key=lambda r: r[7]):
        lines.append(f"| `{name}` | {vocab} | {params / 1e6:.2f}M | {share:.0%} | {cpt:.2f} | ~{ctx:.0f} | {vt:.3f} | **{vc:.3f}** | {secs:.0f}s |")
    lines += ["", f"## Ek sentence ke tokens", "", f"`{SENTENCE}` ({len(SENTENCE)} characters)", ""]
    for name, n, split in splits:
        lines.append(f"- **{name}** ({n} tokens): `{split}`")
    lines += ["", "## Har model ka likha hua text (temperature 0.8)", ""]
    for name, text in samples:
        lines += [f"**{name}**", "", "```", text, "```", ""]
    report = "\n".join(lines)
    (OUT / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
