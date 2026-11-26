# Tokenizers (1500 steps each, mps, model: 4 layers, d=128, context 64 tokens)

| Tokenizer | Vocab | Params | Embeddings ka hissa | Chars/token | Context (chars) | Val loss / token | **Val loss / char** | Waqt |
|---|---|---|---|---|---|---|---|---|
| `gpt2` | 50257 | 13.72M | 94% | 3.09 | ~198 | 4.817 | **1.557** | 608s |
| `bpe1024` | 1024 | 1.06M | 25% | 2.32 | ~148 | 3.647 | **1.572** | 31s |
| `bpe512` | 512 | 0.93M | 14% | 2.06 | ~132 | 3.304 | **1.607** | 30s |
| `bpe256` | 256 | 0.87M | 8% | 1.78 | ~114 | 2.941 | **1.649** | 31s |
| `char` | 65 | 0.82M | 2% | 1.00 | ~64 | 1.770 | **1.770** | 32s |

## Ek sentence ke tokens

`ROMEO: But soft, what light through yonder window breaks?` (57 characters)

- **char** (57 tokens): `R|O|M|E|O|:| |B|u|t| |s|o|f|t|,| |w|h|a|t| |l|i|g|h|t| |t|h|r|o|u|g|h| |y|o|n|d|e|r| |w|i|n|d|o|w| |b|r|e|a|k|s|?`
- **bpe256** (34 tokens): `R|O|M|E|O|:| |B|ut| so|f|t|,| what| li|ght| th|r|ou|gh| y|o|nd|er| w|in|d|ow| b|re|a|k|s|?`
- **bpe512** (28 tokens): `R|OM|EO|:| |But| so|f|t|,| what| li|ght| th|r|ou|gh| y|ond|er| w|ind|ow| b|re|ak|s|?`
- **bpe1024** (22 tokens): `ROMEO|:| |But| so|ft|,| what| li|ght| thr|ough| y|ond|er| w|ind|ow| bre|ak|s|?`
- **gpt2** (14 tokens): `ROM|EO|:| But| soft|,| what| light| through| y|onder| window| breaks|?`

## Har model ka likha hua text (temperature 0.8)

**char**

```
Which shall me in shall juch this to the were,
Thou shall have soull heop I am duke this streath,
His son so swortter and the pats argate
Than you thou the heart tofe on the natter's the king.

ANGELO:
Go she the presul on; sell datch upon the sound.
```

**bpe256**

```
What'st thou warble am.

HENRY BOLINGBROKE:
Well, as he was?

CORIOLANUS:
Claughter! O if it not what is help me.

CAMARGUMBILKE:
This holed be at the tremble.

CLARENCE:
Alas, good good blood, attend of this conss,
And with the ra
```

**bpe512**

```
For Pompey thus; and, if those the people,
And Englands of Buckinghambrokeed dies,
Thy conside to his hide,
Which is at the bawdeed expose withined bawful
From beseech the tale, captaintledge me
That thou canst rugged: by my crown
Richard by quest
```

**bpe1024**

```
For that death is set down; and, proclast,
So many linning, I pray himself to the king,
That murder, with baimship of heavens.

ANGELO:
There's by the prevolt, that's one sight.

QUEEN ELIZABETH:
Ifk you, sir, friar, that's good York is,
After the way to the general titleler.
```

**gpt2**

```
AUFIDIUS:
What art thou, sir?

HERMIONE:
Give me; for the very good good sir?

First Murderer:
The law of repro is covets that
Than so, and my sounds, I pray you.

Messenger:
Thou didst as ever:
That's love, not what I
```
