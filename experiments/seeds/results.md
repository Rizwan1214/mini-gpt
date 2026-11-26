# Experiments (1500 steps each, mps, 3 seeds (1337–1339), mean ± std)

| Experiment | Kya badla | Params | Train loss | Val loss | Val − Train | vs baseline | Pakka? | Waqt |
|---|---|---|---|---|---|---|---|---|
| `no_dropout` | dropout 0.1 → 0 | 0.82M | 1.540 | **1.730** ± 0.001 | +0.191 | -0.062 | haan | 102s |
| `layers_6` | 4 → 6 layers | 1.21M | 1.598 | **1.761** ± 0.007 | +0.163 | -0.031 | haan | 138s |
| `baseline` | 4 layers, d=128, context 64, dropout 0.1, lr 1e-3 | 0.82M | 1.624 | **1.793** ± 0.006 | +0.169 | – | – | 104s |
| `lr_high` | learning rate 1e-3 → 1e-2 | 0.82M | 1.635 | **1.813** ± 0.009 | +0.178 | +0.020 | shayad shor | 101s |
