"""Step 1: Bigram model, sab se simple language model. Isme attention nahi hai.

Har character sirf apne aap ko dekh kar agla character guess karta hai.
Iska loss "baseline" hai: GPT ko is se kaafi behtar karna chahiye,
kyunki attention ki madad se wo pichle characters bhi dekh sakta hai.

Chalao:  python3 bigram.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import CharData, pick_device

torch.manual_seed(1337)


class Bigram(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # Ek table: row = current character, us row ke numbers = agle character ke logits
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss


def main():
    data = CharData()
    device = pick_device()
    model = Bigram(data.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    print(f"vocab = {data.vocab_size} characters, device = {device}")
    print(f"Random guess ka loss = ln({data.vocab_size}) = {torch.log(torch.tensor(float(data.vocab_size))):.3f}")
    start = time.time()
    for step in range(3001):
        x, y = data.get_batch("train", 32, 8, device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                val = sum(model(*data.get_batch("val", 32, 8, device))[1].item() for _ in range(50)) / 50
            print(f"step {step:5d}  train {loss.item():.3f}  val {val:.3f}  ({time.time() - start:.0f}s)")

    # Sample: ek character se shuru karo aur aage likhwao
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(300):
            logits, _ = model(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    print("\n--- Bigram sample ---")
    print(data.decode(idx[0].tolist()))


if __name__ == "__main__":
    main()
