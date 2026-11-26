"""Data: text ko tokens (numbers) mein badalna, aur training ke liye batches banana.

Tokenizer badla ja sakta hai (dekho tokenization.py): "char" (default), "bpe512", "gpt2".
Model ka kaam: har position par agla token predict karna.
"""

from pathlib import Path

import torch

from tokenization import CharTokenizer, make_tokenizer

DATA_FILE = Path(__file__).parent / "data" / "input.txt"


class TextData:
    def __init__(self, tokenizer="char", path: Path = DATA_FILE, val_fraction: float = 0.1):
        self.text = path.read_text(encoding="utf-8")
        # Pehle text ko train/val mein baanto, phir dono ko alag encode karo.
        # Is tarah val ke characters ki asli ginti pata rehti hai (tokenizers compare karne ke liye zaroori).
        n = int((1 - val_fraction) * len(self.text))
        train_text, val_text = self.text[:n], self.text[n:]

        # Tokenizer sirf train text se seekhta hai (val ko "dekhna" cheating hogi)
        self.tokenizer = make_tokenizer(tokenizer, train_text) if isinstance(tokenizer, str) else tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        # Train par seekhta hai; val (jo model ne kabhi nahi dekha) se check karte hain ke ratta to nahi laga raha
        self.train = torch.tensor(self.encode(train_text), dtype=torch.long)
        self.val = torch.tensor(self.encode(val_text), dtype=torch.long)
        # Ek token mein average kitne characters: loss ko "per character" mein badalne ke liye
        self.val_chars_per_token = len(val_text) / len(self.val)

    def encode(self, s: str) -> list[int]:
        return self.tokenizer.encode(s)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def chars(self) -> list[str]:
        """Character tokenizer ki vocab (purane code ke liye)."""
        return self.tokenizer.vocab if isinstance(self.tokenizer, CharTokenizer) else []

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str):
        """Random jagah se `batch_size` tukre uthao.

        x = tokens ka tukra, y = wahi tukra ek token aage shift kiya hua.
        Yani har position par y[t] wo token hai jo x[t] ke baad aata hai.
        """
        src = self.train if split == "train" else self.val
        ix = torch.randint(len(src) - block_size - 1, (batch_size,))
        x = torch.stack([src[i : i + block_size] for i in ix])
        y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)


# Purana naam: bigram.py aur pehle ka code CharData use karta hai
CharData = TextData


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Mac GPU
        return "mps"
    return "cpu"
