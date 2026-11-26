"""Step 3: Fine-tuning. Shuru se train karne ki jagah, pehle se trained GPT-2 lo aur usay apne data par thora aur sikhao.

GPT-2 small (124M parameters) ne internet ka ~40GB text padha hua hai: grammar, lafz, duniya ki baatein.
Hum usay sirf kuch sau steps Shakespeare dikhayenge aur dekhenge:
  1. Fine-tuning se PEHLE wo is text par kitna acha hai (zero-shot)
  2. BAAD mein kitna behtar hua, aur uska andaaz kaise badla

    python3 finetune.py                       (400 steps, sirf aakhri 4 layers train hoti hain)
    python3 finetune.py --train-layers 12     (poora model; 8 GB Mac par bohat dheema, memory kam padti hai)
    python3 finetune.py --steps 50            (jaldi test)
    python3 finetune.py --save          (fine-tuned model finetuned-gpt2/ mein save karo, ~500 MB)

Nateeje: experiments/finetune/results.md aur losses.csv
"""

import argparse
import csv
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

from data import TextData, pick_device

OUT = Path(__file__).parent / "experiments" / "finetune"
PROMPT = "ROMEO:"


def total_ram_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3


def load_gpt2(device: str) -> GPT2LMHeadModel:
    try:  # pehle se download hua ho to internet ke baghair
        model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    return model.to(device)


def freeze_all_but_last(model: GPT2LMHeadModel, n_layers: int) -> int:
    """Partial fine-tuning: sirf aakhri `n_layers` blocks (aur aakhri LayerNorm) seekhenge.

    Pehli layers aur embeddings "freeze": unke gradients aur Adam ki memory nahi banti.
    Neeche ki layers aam baatein jaanti hain (lafz, grammar), jo Shakespeare ke liye bhi kaam aati hain;
    upar ki layers andaaz aur pattern ko adapt karti hain. Wapas: kitne parameters train honge.
    """
    blocks = model.transformer.h
    for p in model.parameters():
        p.requires_grad = False
    for block in blocks[len(blocks) - n_layers :]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.transformer.ln_f.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lm_loss(model, x, y):
    """Har position par agle token ka cross-entropy (bilkul gpt.py jaisa)."""
    logits = model(x).logits
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def evaluate(model, data, block, batch, device, iters=40) -> float:
    model.eval()  # dropout band
    torch.manual_seed(0)  # har baar wahi val batches, taake pehle/baad ka muqabla insaaf se ho
    losses = [lm_loss(model, *data.get_batch("val", batch, block, device)).item() for _ in range(iters)]
    model.train()
    return statistics.fmean(losses)


@torch.no_grad()
def sample(model, data, device, n_tokens=80, seed=42) -> str:
    model.eval()
    torch.manual_seed(seed)
    ids = torch.tensor([data.encode(PROMPT)], device=device)
    # repetition_penalty: pehle aa chuke tokens thore kam pasand kiye jaate hain. Iske baghair chhote
    # models aksar ek hi line baar baar dohrane ke loop mein phans jaate hain ("CISABELLA: CISABELLA: ...").
    out = model.generate(ids, max_new_tokens=n_tokens, do_sample=True, temperature=0.8, top_k=50,
                         repetition_penalty=1.2, pad_token_id=50256)
    model.train()
    return data.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-5, help="fine-tuning mein LR chhota rakhte hain, warna purana ilm mit jata hai")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--train-layers", type=int, default=4, help="GPT-2 ke 12 blocks mein se aakhri kitne train hon")
    ap.add_argument("--block", type=int, default=128, help="context, tokens mein")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--device", default=None, help="cpu, mps ya cuda (default: khud chuno)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device or pick_device()
    if device == "mps" and args.device is None and total_ram_gb() <= 8:
        # 8 GB Mac par GPU aur RAM ek hi memory share karte hain; GPT-2 ki training swap mein chali jati hai
        # aur har step pichle se dheema hota hai. CPU dheema sahi, lekin mustaqil hai.
        print("8 GB Mac: GPU memory kam hai, is liye CPU use ho raha hai (--device mps se badal sakte ho)")
        device = "cpu"
    data = TextData("gpt2")  # GPT-2 ka apna tokenizer; train/val split baaki scripts jaisa
    cpt = data.val_chars_per_token
    model = load_gpt2(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 small: {n_params / 1e6:.0f}M parameters | {cpt:.2f} chars/token | device {device}\n")

    # 1. Fine-tuning se pehle
    before = evaluate(model, data, args.block, args.batch, device)
    before_text = sample(model, data, device)
    print(f"PEHLE (zero-shot): val {before:.3f}/token = {before / cpt:.3f}/char")
    print(before_text, "\n")

    # 2. Fine-tuning: wahi training loop jo gpt.py mein hai, bas model pehle se trained hai
    torch.manual_seed(1337)
    trainable = freeze_all_but_last(model, args.train_layers)
    print(f"Train ho rahe: aakhri {args.train_layers}/12 blocks = {trainable / 1e6:.1f}M of {n_params / 1e6:.0f}M parameters\n")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log = [(0, None, before)]
    # Early stopping: sab se kam val loss wale weights sambhal kar rakho. Training lambi chale to
    # model train data ke mutabiq zyada dhal jata hai aur val loss phir barhne lagta hai.
    trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    best_val, best_step = before, 0
    best_state = {n: p.detach().clone() for n, p in trainable_params.items()}
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = data.get_batch("train", args.batch, args.block, device)
        loss = lm_loss(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # bade jhatkon se bachao
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, data, args.block, args.batch, device)
            log.append((step, loss.item(), val))
            marker = ""
            if val < best_val:
                best_val, best_step = val, step
                best_state = {n: p.detach().clone() for n, p in trainable_params.items()}
                marker = "  <- ab tak sab se acha"
            print(f"step {step:4d}  train {loss.item():.3f}  val {val:.3f}/token = {val / cpt:.3f}/char  ({time.time() - start:.0f}s){marker}")
    secs = time.time() - start
    last = log[-1][2]

    # 3. Fine-tuning ke baad: sab se ache step wale weights wapas lagao
    with torch.no_grad():
        for n, p in trainable_params.items():
            p.copy_(best_state[n])
    # Check: wapas lagaye hue weights sach mein wahi val loss dete hain?
    after = evaluate(model, data, args.block, args.batch, device)
    assert abs(after - best_val) < 1e-3, f"restore failed: {after:.4f} != {best_val:.4f}"
    after_text = sample(model, data, device)
    print(f"\nAakhri step ({args.steps}): val {last / cpt:.3f}/char | sab se acha step {best_step}: val {after / cpt:.3f}/char")
    print(f"BAAD (step {best_step} wala model):")
    print(after_text)

    with open(OUT / "losses.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "train", "val"), *[(s, "" if t is None else t, v) for s, t, v in log]])
    report = f"""# GPT-2 fine-tuning ({args.steps} steps, aakhri {args.train_layers}/12 blocks = {trainable / 1e6:.1f}M params train hue, lr {args.lr}, batch {args.batch} × {args.block} tokens, {device}, {secs:.0f}s)

| | Val loss / token | Val loss / char |
|---|---|---|
| Fine-tuning se pehle (zero-shot) | {before:.3f} | {before / cpt:.3f} |
| Sab se acha step ({best_step}), early stopping ne yahi rakha | {after:.3f} | **{after / cpt:.3f}** |
| Aakhri step ({args.steps}) | {last:.3f} | {last / cpt:.3f} |

## Prompt `{PROMPT}`: pehle

```
{before_text.strip()}
```

## Prompt `{PROMPT}`: baad (step {best_step} wala model)

```
{after_text.strip()}
```
"""
    (OUT / "results.md").write_text(report)
    print(f"\nSaved {OUT / 'results.md'}")

    if args.save:
        model.save_pretrained("finetuned-gpt2")
        data.tokenizer.tok.save_pretrained("finetuned-gpt2")
        print("Saved finetuned-gpt2/")


if __name__ == "__main__":
    main()
