"""Trained model (model.pt) se text likhwao.

    python3 sample.py
    python3 sample.py --prompt "ROMEO:" --temp 0.7 --n 600

temperature: kam (0.5) = mehfooz, dohrane wala text; zyada (1.2) = creative lekin ajeeb.
"""

import argparse

import torch

from data import pick_device
from tokenization import CharTokenizer, load_tokenizer
from gpt import GPT, Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model.pt")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--n", type=int, default=500, help="kitne naye tokens")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    model = GPT(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Naye checkpoints apna tokenizer saath rakhte hain; purane sirf characters ki list
    tok = load_tokenizer(ckpt["tokenizer"]) if "tokenizer" in ckpt else CharTokenizer(ckpt["chars"])
    ids = tok.encode(args.prompt) or tok.encode("\n")
    idx = torch.tensor([ids], device=device)
    out = model.generate(idx, args.n, temperature=args.temp, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
