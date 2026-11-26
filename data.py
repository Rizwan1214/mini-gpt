"""Data: text ko numbers mein badalna, aur training ke liye batches banana.

Har character ek token hai (character-level tokenizer). Model ka kaam:
har position par agla character predict karna.
"""

from pathlib import Path

import torch

DATA_FILE = Path(__file__).parent / "data" / "input.txt"


class CharData:
    def __init__(self, path: Path = DATA_FILE, val_fraction: float = 0.1):
        self.text = path.read_text(encoding="utf-8")
        # Vocabulary: text mein jitne alag characters hain
        self.chars = sorted(set(self.text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}  # character -> number
        self.itos = {i: c for c, i in self.stoi.items()}  # number -> character

        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int((1 - val_fraction) * len(data))
        # Train par seekhta hai; val (jo model ne kabhi nahi dekha) se check karte hain ke ratta to nahi laga raha
        self.train, self.val = data[:n], data[n:]

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        """Random jagah se `batch_size` tukre uthao.

        x = text ka tukra, y = wahi tukra ek character aage shift kiya hua.
        Yani har position par y[t] wo character hai jo x[t] ke baad aata hai.
        """
        src = self.train if split == "train" else self.val
        ix = torch.randint(len(src) - block_size - 1, (batch_size,))
        x = torch.stack([src[i : i + block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Mac GPU
        return "mps"
    return "cpu"
