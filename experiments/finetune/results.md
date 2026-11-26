# GPT-2 fine-tuning (400 steps, aakhri 4/12 blocks = 28.4M params train hue, lr 5e-05, batch 4 × 128 tokens, cpu, 283s)

| | Val loss / token | Val loss / char |
|---|---|---|
| Fine-tuning se pehle (zero-shot) | 4.511 | 1.458 |
| Sab se acha step (150), early stopping ne yahi rakha | 3.693 | **1.194** |
| Aakhri step (400) | 4.001 | 1.293 |

## Prompt `ROMEO:`: pehle

```
ROMEO: 1) This was the name of one more character in A-3 and then again later on, it's actually a "1" (this is an abbreviation for Anakin Skywalker). It would be interesting to find out what he wanted his father after being captured by Darth Vader. 2)(2) As mentioned above, this wasn't very bigoted either but if you're going through ANAG
```

## Prompt `ROMEO:`: baad (step 150 wala model)

```
ROMEO:
CISABELLA : (VOCALOIDED) - pescatore.com, here is the latest news from CINCINNATI!

FRIARIO VIOLANUS CHRISTIANIA- It's a very interesting time of year... what would you think? What does your mother wish for with her birthday tomorrow afternoon...? Have I let myself be tempted
```
