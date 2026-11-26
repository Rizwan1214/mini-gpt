"""Tokenizers: text ko tokens (numbers) mein todne ke teen tareeqe.

1. CharTokenizer: har character ek token (vocab ~65). Simple, lekin text lamba ho jata hai.
2. BPETokenizer:  khud seekha hua Byte-Pair Encoding. Sab se zyada baar saath aane wale
                  do tokens ko baar baar jod kar naya token banata hai ("t"+"h" -> "th",
                  "th"+"e" -> "the" ...). Yahi tareeqa GPT-2/GPT-4 ka hai.
3. GPT2Tokenizer: OpenAI ka asli GPT-2 tokenizer (50,257 tokens), Hugging Face cache se.

    python3 tokenization.py            (BPE train karo aur seekhe hue merges dekho)
"""

from __future__ import annotations

import re
from collections import Counter

# Text ko "words" mein todna (GPT-2 jaisa): har word apni aage wali space ke saath.
# BPE kabhi word ki hadd paar nahi jodta, is liye " the" ek token ban sakta hai lekin "e t" nahi.
WORD_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text)


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str]):
        self.vocab = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "chars": self.vocab}


class BPETokenizer:
    kind = "bpe"

    def __init__(self, base: list[str], merges: list[tuple[str, str]]):
        self.base = base
        self.merges = merges
        # Vocab = shuru ke characters + har merge se bana naya token
        self.vocab = base + [a + b for a, b in merges]
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.rank = {pair: r for r, pair in enumerate(merges)}  # kaunsa merge pehle seekha gaya
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """BPE seekhna: jab tak vocab poora na ho, sab se aam pair ko jodte jao."""
        base = sorted(set(text))
        word_freq = Counter(words_of(text))
        splits = {w: list(w) for w in word_freq}  # har word abhi characters mein toota hua
        merges: list[tuple[str, str]] = []

        while len(base) + len(merges) < vocab_size:
            # 1. Har adjacent pair kitni baar aata hai (word ki frequency ke saath)
            pairs: Counter[tuple[str, str]] = Counter()
            for w, freq in word_freq.items():
                parts = splits[w]
                for a, b in zip(parts, parts[1:]):
                    pairs[(a, b)] += freq
            if not pairs:
                break
            # 2. Sab se aam pair (barabri par alphabetical, taake har baar same nateeja aaye)
            best = max(pairs, key=lambda p: (pairs[p], p))
            merges.append(best)
            if verbose and len(merges) <= 30:
                print(f"merge {len(merges):3d}: {best[0]!r} + {best[1]!r} -> {best[0] + best[1]!r}  ({pairs[best]} baar)")
            # 3. Jahan jahan ye pair hai, usay ek token bana do
            a, b = best
            for w, parts in splits.items():
                if len(parts) < 2:
                    continue
                merged, i = [], 0
                while i < len(parts):
                    if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(parts[i])
                        i += 1
                splits[w] = merged
        return cls(base, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_word(self, word: str) -> list[int]:
        if word not in self._cache:
            parts = [c for c in word if c in self.stoi]
            # Wahi merges usi order mein lagao jis order mein seekhe the
            while len(parts) > 1:
                ranked = [(self.rank.get((a, b), float("inf")), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
                r, i = min(ranked)
                if r == float("inf"):
                    break
                parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2 :]
            self._cache[word] = [self.stoi[p] for p in parts]
        return self._cache[word]

    def encode(self, text: str) -> list[int]:
        return [i for w in words_of(text) for i in self._encode_word(w)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "base": self.base, "merges": [list(m) for m in self.merges]}


class GPT2Tokenizer:
    """OpenAI GPT-2 ka asli tokenizer (byte-level BPE, 50,257 tokens)."""

    kind = "gpt2"

    def __init__(self):
        from transformers import GPT2TokenizerFast  # sirf zaroorat par import

        # Pehle se download hua ho to internet ke baghair chalta hai
        try:
            self.tok = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        except OSError:
            self.tok = GPT2TokenizerFast.from_pretrained("gpt2")

    @property
    def vocab_size(self) -> int:
        return len(self.tok)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def to_dict(self) -> dict:
        return {"kind": self.kind}


def make_tokenizer(spec: str, text: str):
    """'char', 'bpe512' (BPE, vocab 512), ya 'gpt2'."""
    if spec == "char":
        return CharTokenizer.from_text(text)
    if spec.startswith("bpe"):
        return BPETokenizer.train(text, int(spec[3:] or 512))
    if spec == "gpt2":
        return GPT2Tokenizer()
    raise ValueError(f"unknown tokenizer {spec!r} (char, bpe512, gpt2)")


def load_tokenizer(d: dict):
    """Checkpoint mein save hue tokenizer ko wapas banao."""
    if d["kind"] == "char":
        return CharTokenizer(d["chars"])
    if d["kind"] == "bpe":
        return BPETokenizer(d["base"], [tuple(m) for m in d["merges"]])
    if d["kind"] == "gpt2":
        return GPT2Tokenizer()
    raise ValueError(d["kind"])


def show_split(tok, text: str) -> str:
    """Tokens ko | se alag kar ke dikhao, jaise 'ROM|EO|:| But| soft'."""
    return "|".join(tok.decode([i]) for i in tok.encode(text))


if __name__ == "__main__":
    import time
    from pathlib import Path

    text = (Path(__file__).parent / "data" / "input.txt").read_text(encoding="utf-8")
    start = time.time()
    bpe = BPETokenizer.train(text, 512, verbose=True)
    print(f"\nTrained BPE with {bpe.vocab_size} tokens in {time.time() - start:.1f}s")
    print("Kuch lambe tokens:", sorted(bpe.vocab, key=len)[-12:])
    sample = "ROMEO: But soft, what light through yonder window breaks?"
    for tok in (CharTokenizer.from_text(text), bpe):
        ids = tok.encode(sample)
        assert tok.decode(ids) == sample, "decode(encode(x)) != x"
        print(f"\n{tok.kind:5} {len(ids):3d} tokens: {show_split(tok, sample)}")
