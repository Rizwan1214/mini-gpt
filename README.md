# mini-GPT

Shuru se banaya hua ek chhota GPT (decoder-only transformer), character-level, PyTorch mein.
Ye seekhne ke liye hai: har line ke saath Roman Urdu comments hain.

## Files

| File | Kya karti hai |
|---|---|
| `data.py` | Text → numbers (character tokenizer), train/val split, batches (`x` aur ek aage shift kiya `y`) |
| `bigram.py` | Step 1: sab se simple model, attention ke baghair. Iska loss baseline hai |
| `gpt.py` | Step 2: asli transformer (embeddings → 4 blocks [LayerNorm → causal multi-head attention → MLP, residuals] → logits) aur training loop |
| `sample.py` | Trained `model.pt` se text likhwana |
| `data/input.txt` | Tiny Shakespeare (~1.1 MB), [karpathy/char-rnn](https://github.com/karpathy/char-rnn) se |

## Chalana

```bash
pip install -r requirements.txt
python3 bigram.py            # baseline, ~5 second
python3 gpt.py               # GPT training, 3000 steps
python3 sample.py --prompt "ROMEO:" --temp 0.8
```

Mac par khud Apple GPU (`mps`) use hota hai, NVIDIA par `cuda`, warna CPU.

## Pehli training ke nateeje (Mac, MPS)

| Model | Parameters | Val loss | Waqt |
|---|---|---|---|
| Random guess | – | 4.17 (ln 65) | – |
| Bigram | 4K | ~2.49 | ~5s |
| mini-GPT (4 layers, 4 heads, d=128) | 0.82M | **1.665** | ~65s, 3000 steps |

GPT ka text (`--prompt "ROMEO:" --temp 0.7`):

```
ROMEO:
Then, and good this and heart, their states stoods and my brother,
Nor wondren the recome, burt and the hath house.

GREGORY:
I had come of the hunds and all the cousin.
```

Lafz zyada tar asli hain aur drama ka format (naam, colon, lines) seekh liya hai, lekin matlab abhi nahi.
Zyada steps aur bada model is ko aur behtar karte hain.

## Apne experiments

`gpt.py` mein `Config` badlo aur dekho loss par kya asar hota hai:

- `n_layers` ya `d_model` barhao: model bara, loss kam, lekin dheema
- `block_size` barhao: zyada pichla context dikhta hai
- `dropout = 0`: train loss aur val loss ka farq (overfitting) barhta dikhega
- `--lr 3e-3` ya `--lr 1e-4`: bohat zyada ya bohat kam learning rate ka asar
- Apna text `data/input.txt` mein daalo (jaise Roman Urdu shayari), aur model usi ke andaaz mein likhega

`losses.csv` mein har eval ka train/val loss save hota hai.
