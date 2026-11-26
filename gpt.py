"""Step 2: Asli mini-GPT (decoder-only transformer), shuru se.

Bigram se farq: har character attention ki madad se pichle `block_size`
characters ko dekh sakta hai.

Chalao:   python3 gpt.py                      (default: 3000 steps)
          python3 gpt.py --steps 500          (jaldi test)
Phir:     python3 sample.py                   (trained model se text likhwao)
"""

import argparse
import csv
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CharData, pick_device


@dataclass
class Config:
    block_size: int = 64  # kitne pichle characters dekh sakta hai (context)
    d_model: int = 128  # har token ka vector kitna lamba hai
    n_heads: int = 4  # attention heads
    n_layers: int = 4  # kitne transformer blocks
    dropout: float = 0.1  # training mein random neurons band: ratta lagane se rokta hai
    vocab_size: int = 0  # data se bharta hai
    use_attention: bool = True  # False = attention hata do (experiment: iske baghair kya hota hai?)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # Q, K, V teeno ek saath
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)  # W_O: heads ko aapas mein milata hai
        self.drop = nn.Dropout(cfg.dropout)
        # Causal mask: position t sirf 0..t dekh sakti hai (lower triangle)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).bool()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        h, dh = self.cfg.n_heads, C // self.cfg.n_heads
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, d_head): har head apne hisse ke columns leta hai
        q, k, v = [t.view(B, T, h, dh).transpose(1, 2) for t in (q, k, v)]

        att = (q @ k.transpose(-2, -1)) / dh**0.5  # scores: (B, h, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))  # future -> -inf
        att = self.drop(F.softmax(att, dim=-1))  # har row ka total 1

        out = att @ v  # weighted sum of values: (B, h, T, dh)
        out = out.transpose(1, 2).reshape(B, T, C)  # heads concat
        return self.drop(self.proj(out))


class Block(nn.Module):
    """Ek transformer block: attention + MLP, dono ke saath residual aur LayerNorm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.use_attention = cfg.use_attention
        if cfg.use_attention:  # attention ke baghair wale experiment mein ye parameters bante hi nahi
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        if self.use_attention:
            x = x + self.attn(self.ln1(x))  # residual: purani info bachi rehti hai
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # token -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)  # position -> vector (seekha hua)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)  # vector -> har character ka score

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """Ek ek character aage likho: predict -> sample -> jodo -> dohrao."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])  # context ki hadd
            logits = logits[:, -1, :] / temperature  # sirf aakhri position ka agla character
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, data, cfg, batch_size, device, iters=50):
    """Kai batches ka average loss: ek batch ka loss bohat shor wala hota hai."""
    model.eval()  # dropout band
    out = {}
    for split in ("train", "val"):
        losses = [model(*data.get_batch(split, batch_size, cfg.block_size, device))[1].item() for _ in range(iters)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def train_model(cfg: Config, data: CharData, steps: int, lr: float = 1e-3, batch: int = 32,
                eval_every: int = 250, seed: int = 1337, device: str | None = None, verbose: bool = True):
    """Ek model train karo. Wapas deta hai: (model, log [(step, train, val)], seconds)."""
    torch.manual_seed(seed)
    device = device or pick_device()
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    log = []
    start = time.time()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            est = estimate_loss(model, data, cfg, batch, device)
            log.append((step, est["train"], est["val"]))
            if verbose:
                print(f"step {step:5d}  train {est['train']:.3f}  val {est['val']:.3f}  ({time.time() - start:.0f}s)")
        if step == steps:
            break

        # Training ka asli loop: batch -> forward -> loss -> backward -> update
        x, y = data.get_batch("train", batch, cfg.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return model, log, time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    data = CharData()
    device = pick_device()
    cfg = Config(vocab_size=data.vocab_size)
    n_params = sum(p.numel() for p in GPT(cfg).parameters())
    print(f"{n_params / 1e6:.2f}M parameters | vocab {data.vocab_size} | device {device}")
    print(f"config: {asdict(cfg)}")

    model, log, _ = train_model(cfg, data, args.steps, args.lr, args.batch, args.eval_every, device=device)

    torch.save({"config": asdict(cfg), "model": model.state_dict(), "chars": data.chars}, args.out)
    with open("losses.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "train", "val"), *log])
    print(f"\nSaved {args.out} and losses.csv")

    model.eval()
    idx = torch.tensor([data.encode("\n")], device=device)
    print("\n--- GPT sample ---")
    print(data.decode(model.generate(idx, 500, temperature=0.8)[0].tolist()))


if __name__ == "__main__":
    main()
