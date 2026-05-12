# OpenFold 2 Baseline (Unlimited Track)

## Overview

Full OpenFold 2 model trained on the unlimited track. Uses the OpenFold implementation from `/home/ebrahim/openfold` with a DDP training script (`train_ddp.py`).

- **Model**: 92.9M parameters
- **Track**: Unlimited
- **Submission**: `submissions/openfold_unlimited/`

## Architecture

| Hyperparameter | Value |
|---|---|
| MSA depth (main stack) | 128 |
| Extra MSA depth | 1024 |
| Recycling iterations | 3 (4 total forward passes) |

## Training Setup

| Setting | Value |
|---|---|
| Effective batch size | 128 (4 GPUs × 1 sample × 32 grad accum) |
| Crop size | 256 residues |
| MSA depth (input) | 512 |
| Train crop mode | random |
| Val crop mode | center |
| Train MSA sample mode | random |
| Val MSA sample mode | top |
| Precision | BF16 |
| Max steps | 6000 |
| Grad accum steps | 32 |
| Log every | 10 steps |
| Eval every | 500 steps |
| Save every | 500 steps |
| Eval foldscore components | true |

## Optimizer

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.0 |
| β1 | 0.9 |
| β2 | 0.99 |
| ε | 1e-6 |
| Grad clip norm | 0.1 |
| LR decay factor | 0.95 |

## Learning Rate Schedule

Three phases:
1. **Warmup**: linear ramp over steps 0→67
2. **Decay**: exponential decay (factor 0.95) starting at step 3652
3. **Finetune**: LR scaled by 0.5 starting at step 5217, ramping over 500 steps

## Finetune Phase Config Changes (step 5217+)

Matching OpenFold's finetuning preset (AF2 Suppl. Table 4):

| Parameter | Initial | Finetune |
|---|---|---|
| crop_size | 256 | 384 |
| model.msa_depth | 128 | 512 |
| model.extra_msa_depth | 1024 | 5120 |
| violation loss weight | 0.0 | 1.0 |
| experimentally_resolved weight | 0.0 | 0.01 |

Note: `extra_msa_depth=5120` is bounded in practice by `data.msa_depth=512` (the number of MSA sequences loaded at collation time). The full 5120 would require more sequences in the preprocessed data.

## Training Run

Started from scratch (step 0). ETA ~79 hours on 4 GPUs.

### Crash & Fix (2026-05-10)

The first run crashed at step 500 during validation with an NCCL `ALLREDUCE` timeout.

**Root cause**: Validation runs only on rank 0 while ranks 1–3 wait at a `dist.barrier()`. With `eval_foldscore_components: true`, the 1000-sample val set takes ~33 min at ~2s/it. Without foldscore it still takes ~15 min. Both exceed PyTorch's default 10-minute NCCL watchdog timeout.

**Fixes applied**:
1. Raised NCCL timeout to 2 hours in `init_process_group`
2. Moved checkpoint save to happen **before** eval — so a future eval crash doesn't lose the step's training progress

**Data lost**: Step 0 was the only checkpoint; the ~6 hours of training to step 500 was lost.

## Val Metrics

| Step | val_loss | val_lddt_ca | val_rmsd_ca |
|---|---|---|---|
| — | — | — | — |

*(to be filled as training progresses)*
