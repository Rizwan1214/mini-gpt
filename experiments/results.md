# Experiments (1500 steps each, mps)

| Experiment | Kya badla | Params | Train loss | Val loss | Val − Train | vs baseline | Waqt |
|---|---|---|---|---|---|---|---|
| `no_dropout` | dropout 0.1 → 0 | 0.82M | 1.525 | **1.731** | +0.206 | -0.065 | 36s |
| `layers_6` | 4 → 6 layers | 1.21M | 1.609 | **1.769** | +0.160 | -0.028 | 49s |
| `baseline` | 4 layers, d=128, context 64, dropout 0.1, lr 1e-3 | 0.82M | 1.615 | **1.796** | +0.182 | – | 40s |
| `lr_high` | learning rate 1e-3 → 1e-2 | 0.82M | 1.636 | **1.837** | +0.201 | +0.040 | 33s |
| `layers_2` | 4 → 2 layers | 0.42M | 1.715 | **1.870** | +0.155 | +0.074 | 25s |
| `context_16` | context 64 → 16 characters | 0.81M | 1.823 | **1.958** | +0.135 | +0.162 | 20s |
| `lr_low` | learning rate 1e-3 → 1e-4 | 0.82M | 2.287 | **2.305** | +0.019 | +0.509 | 35s |
| `no_attention` | attention hata di (sirf MLP) | 0.55M | 2.469 | **2.496** | +0.027 | +0.699 | 22s |
